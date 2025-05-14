import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from collections import deque
import os
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.font_manager as fm

# ----------------- Helpers -----------------
def reshape_text(txt):
    return get_display(arabic_reshaper.reshape(str(txt)))

def create_db_engine():
    server = "52.48.117.197"
    database = "R1029"
    username = "sa"
    password = "Argus@NEG"
    driver = "ODBC Driver 17 for SQL Server"
    odbc = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};TrustServerCertificate=Yes;"
    )
    url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"
    try:
        eng = create_engine(url, connect_args={"timeout": 5})
        with eng.connect():
            pass
        return eng, None
    except Exception as e:
        return None, str(e)

def convert_gold(cur, amt):
    if cur == 2:   return amt * 6.0 / 7.0
    if cur == 3:   return amt
    if cur == 4:   return amt * 24.0 / 21.0
    if cur == 14:  return amt * 14.0 / 21.0
    return amt
    
PRIORITY_FIDS = {3001, 3100, 3108, 3113, 3104}
def process_fifo(debits, credits, as_of, priority_fids=PRIORITY_FIDS):
    # فلترنالكريدتس حسب التاريخ
    credits = [c for c in credits if c["date"] <= as_of]

    # فرز الديبتس ذوي الأولوية أولاً، بناءً على (date, invoiceref)
    pri = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))
    # ثم بقية الديبتس بنفس القاعدة
    reg = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] not in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))

    excess = 0.0
    for cr in sorted(credits, key=lambda x: (x["date"], x.get("invoiceref", ""))):
        rem = cr["amount"]

        # دَين الأولوية
        while rem > 0 and pri:
            d = pri[0]
            ap = min(rem, d["remaining"])
            d["remaining"] -= ap
            rem -= ap
            if d["remaining"] <= 0:
                d["paid_date"] = cr["date"]
                pri.popleft()

        # بعد نفاد الأولوية ننتقل للعادي
        while rem > 0 and not pri and reg:
            d = reg[0]
            ap = min(rem, d["remaining"])
            d["remaining"] -= ap
            rem -= ap
            if d["remaining"] <= 0:
                d["paid_date"] = cr["date"]
                reg.popleft()

        excess += rem

    remaining = list(pri) + list(reg)
    total_remaining = sum(d["remaining"] for d in remaining)
    net_balance = total_remaining - excess
    return remaining, net_balance


def bucketize(days, grace, length):
    if days <= grace:
        return None
    adj = days - grace
    if adj <= length:
        return f"{grace + 1}-{grace + length}"
    if adj <= 2 * length:
        return f"{grace + length + 1}-{grace + 2 * length}"
    if adj <= 3 * length:
        return f"{grace + 2 * length + 1}-{grace + 3 * length}"
    return f">{grace + 3 * length}"

def format_number(value):
    try:
        value = float(value)
        if value < 0:
            return f"({abs(value):,.2f})"
        elif value == 0:
            return "-"
        else:
            return f"{value:,.2f}"
    except (ValueError, TypeError):
        return str(value)

# ----------------- Data Fetching -----------------
@st.cache_data(ttl=600)
def get_salespersons(_engine):
    return pd.read_sql("SELECT recordid, name,spRef FROM sasp ORDER BY name", _engine)

@st.cache_data(ttl=600)
def get_customers(_engine, sp_id):
    if sp_id is None:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid,reference, keyWords, COALESCE(sasp.name, 'غير محدد') AS sp_name
            FROM fiacc acc
            LEFT JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.groupid = 1
            ORDER BY acc.name
            """
        return pd.read_sql(text(sql), _engine)
    else:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid, reference, keyWords,sasp.name AS sp_name
            FROM fiacc acc
            JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.spid = :sp
            ORDER BY acc.name
            """
        return pd.read_sql(text(sql), _engine, params={"sp": sp_id})

@st.cache_data(ttl=300)
def get_overdues(_engine, sp_id, as_of, grace, length):
    # pull functionid in SQL
    base_sql = """
        SELECT f.accountid,
               f.functionid,
               acc.reference AS code,
               acc.name AS name,
               f.currencyid,
               f.amount,
               f.date,
               COALESCE(f.reference, CAST(f.date AS VARCHAR)) AS invoiceref,
               acc.spid,
               COALESCE(sasp.name,'غير محدد') AS sp_name
        FROM fitrx f
        JOIN fiacc acc ON f.accountid = acc.recordid
        LEFT JOIN sasp ON acc.spid = sasp.recordid
        WHERE acc.groupid = 1 AND f.date <= :as_of
    """
    params = {"as_of": as_of}
    if sp_id is not None:
        base_sql += " AND acc.spid = :sp"
        params["sp"] = sp_id
    base_sql += " ORDER BY acc.reference, f.date"

    raw = pd.read_sql(text(base_sql), _engine, params=params)

    buckets = [
        f"{grace + 1}-{grace + length}",
        f"{grace + length + 1}-{grace + 2 * length}",
        f"{grace + 2 * length + 1}-{grace + 3 * length}",
        f">{grace + 3 * length}"
    ]
    summary_rows = []
    invoice_data = []

    for acc, grp in raw.groupby("accountid"):
        code = grp["code"].iat[0]
        name = grp["name"].iat[0]
        sp_name = grp["sp_name"].iat[0]

        cash_debits = []
        cash_credits = []
        gold_debits = []
        gold_credits = []

        for _, r in grp.iterrows():
            dt = pd.to_datetime(r["date"])
            amt = r["amount"]
            invoiceref = r["invoiceref"]
            fid = r["functionid"]

            # cash
            if r["currencyid"] == 1:
                if amt > 0:
                    cash_debits.append({
                        "date": dt,
                        "remaining": amt,
                        "paid_date": None,
                        "original_amount": amt,
                        "invoiceref": invoiceref,
                        "functionid": fid
                    })
                else:
                    cash_credits.append({"date": dt, "amount": abs(amt)})
            # gold
            else:
                grams = convert_gold(r["currencyid"], amt)
                if amt > 0:
                    gold_debits.append({
                        "date": dt,
                        "remaining": grams,
                        "paid_date": None,
                        "original_amount": grams,
                        "invoiceref": invoiceref,
                        "functionid": fid
                    })
                else:
                    gold_credits.append({"date": dt, "amount": abs(grams)})

        # apply priority-aware FIFO
        pc, net_cash = process_fifo(cash_debits, cash_credits, pd.to_datetime(as_of))
        pg, net_gold = process_fifo(gold_debits, gold_credits, pd.to_datetime(as_of))

        # bucket and build summary/detail rows
        sums = {f"cash_{b}": 0.0 for b in buckets}
        sums.update({f"gold_{b}": 0.0 for b in buckets})
        inv_over = {}

        for drv, net, pfx in [(pc, net_cash, "cash"), (pg, net_gold, "gold")]:
            for d in drv:
                if d["remaining"] > 0:
                    days = ((d.get("paid_date") or pd.to_datetime(as_of)) - d["date"]).days
                    bucket = bucketize(days, grace, length)
                    if bucket:
                        sums[f"{pfx}_{bucket}"] += d["remaining"]
                        ref = d["invoiceref"]
                        if ref not in inv_over:
                            inv_over[ref] = {
                                "Customer Reference": code,
                                "Customer Name": name,
                                "Invoice Ref": ref,
                                "Invoice Date": d["date"].date(),
                                "Overdue G21": 0.0,
                                "Overdue EGP": 0.0,
                                "Delay Days": max(0, days - grace)
                            }
                        inv_over[ref][f"Overdue {'G21' if pfx=='gold' else 'EGP'}"] += d["remaining"]

        # add detail rows
        invoice_data.extend(inv_over.values())

        cash_total = sum(sums[f"cash_{b}"] for b in buckets)
        gold_total = sum(sums[f"gold_{b}"] for b in buckets)
        if cash_total > 0 or gold_total > 0:
            summary_rows.append({
                "AccountID": acc,
                "Customer": name,
                "Code": code,
                "sp_name": sp_name,
                "total_cash_due": net_cash,
                "total_gold_due": net_gold,
                **sums,
                "cash_total": cash_total,
                "gold_total": gold_total
            })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(invoice_data)
    if not detail_df.empty:
      detail_df.sort_values(
        by=["Invoice Date", "Invoice Ref"],
        key=lambda col: col.astype(str),   # نضمن أنها نصوص قابلة للفرز أبجدياً
        inplace=True
    )
    return summary_df, buckets, detail_df

# ----------------- PDF Functions -----------------
def truncate_text(pdf, text, width):
    """Truncate text to fit within specified width."""
    ellipsis = "..."
    text = str(text)
    while pdf.get_string_width(ellipsis + text) > width and text:
        text = text[1:]
    return ellipsis + text if pdf.get_string_width(ellipsis + text) <= width else text

def draw_parameters_table(pdf, params, table_width, col_widths):
    """Draw the parameters table using a dictionary of parameters."""
    # Define parameters with bilingual labels
    parameters = [
        ("Sales Person ", params.get("Sales Person", "")),
        ("Customer ", params.get("Customer", "")),
        ("Due Date ", params.get("Due Date", "")),
        ("Grace Period ", params.get("Grace Period", "")),
        ("Period Length ", params.get("Period Length", ""))
    ]
    
    # Draw header
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(col_widths[0] + col_widths[1], 8, reshape_text("Parameters"), border=1, align="C", fill=True, ln=1)
    pdf.ln(2)

    # Draw parameter rows
    for label, value in parameters:
        pdf.set_fill_color(255, 255, 255)  # White background for rows
        pdf.cell(col_widths[0], 8, reshape_text(label), border=1, align="C", ln=0)
        pdf.cell(col_widths[1], 8, reshape_text(value), border=1, align="C", ln=1)
def draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_widths):
    """Draw the table headers with sub-headers for Balance, Buckets, and Total Delay."""
    row_h = 7
    sub_w_g21, sub_w_egp = sub_widths

    # Main headers
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(name_w, row_h, reshape_text("Customer"), border=1, align="C", fill=True)
    pdf.cell(bal_w, row_h, reshape_text("Balance"), border=1, align="C", fill=True)

    for i, b in enumerate(buckets):
        if i == len(buckets) - 1 and b.startswith(">"):  # Last bucket, e.g., ">75"
            number = b[1:]  # Extract the number after ">"
            display_label = f" Greater Than {number}"  # Replace ">" with "Bigger Than"
        else:
            start, end = b.split("-")
            display_label = f"From {start} to {end}"
        pdf.cell(bucket_w, row_h, reshape_text(display_label), border=1, align="C", fill=True)

    pdf.cell(tot_w, row_h, reshape_text("Total Delay"), border=1, align="C", fill=True)
    pdf.ln(row_h)

    # Sub-headers
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(name_w, row_h, "", border=1, fill=True)  # Empty cell under Customer

    # Balance sub-headers
    pdf.cell(sub_w_g21, row_h, reshape_text("G21"), border=1, align="C", fill=True)
    pdf.cell(sub_w_egp, row_h, reshape_text("EGP"), border=1, align="C", fill=True)

    # Bucket sub-headers
    for _ in buckets:
        pdf.cell(sub_w_g21, row_h, reshape_text("G21"), border=1, align="C", fill=True)
        pdf.cell(sub_w_egp, row_h, reshape_text("EGP"), border=1, align="C", fill=True)

    # Total Delay sub-headers
    pdf.cell(sub_w_g21, row_h, reshape_text("G21"), border=1, align="C", fill=True)
    pdf.cell(sub_w_egp, row_h, reshape_text("EGP"), border=1, align="C", fill=True)

    pdf.ln(row_h)

def build_summary_pdf(df, sp_name, as_of, buckets, selected_customer, grace, length):
    if not isinstance(buckets, (list, tuple)):
        st.error("Invalid buckets parameter: must be a list or tuple")
        return b""

    pdf = FPDF(orientation="L", unit="mm", format="A3")
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 10)

    pdf.cell(0, 5, reshape_text("New Egypt Gold |تقرير اجمالى اعمار الديون"), ln=0, align="C")

    pdf.ln(5)
    pdf.cell(0, 5, f"Execution Date: {datetime.now().strftime('%d/%m/%Y')}", ln=0, align="L")
    pdf.ln(10)

    params = {
        "Sales Person": sp_name,
        "Customer": selected_customer,
        "Due Date": as_of.strftime('%d/%m/%Y'),
        "Grace Period": f"{grace} يوم",
        "Period Lenght ": f"{length} يوم"
    }
    draw_parameters_table(pdf, params, 120, [40, 80])
    pdf.ln(10)

    # Column widths
    name_w, bal_w, bucket_w, tot_w, sub_w_g21, sub_w_egp = 50, 60, 60, 60, 25, 35
    grouped = df.groupby("sp_name", dropna=False) if sp_name == "All" else [(sp_name, df)]

    for sp_id, group in grouped:
        sp_display_name = group["sp_name"].iloc[0] if sp_name == "All" else sp_name
        if sp_id in (0, '0', None):
            sp_display_name = ""  # Empty string instead of "غير محدد"

        pdf.cell(0, 5, reshape_text(f"Sales Person: {sp_display_name}"), ln=1, align="L")
        pdf.ln(4)
        draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, [sub_w_g21, sub_w_egp])

        # Totals
        totals = {
            "total_gold_due": group["total_gold_due"].sum(),
            "total_cash_due": group["total_cash_due"].sum(),
            "gold_total": group["gold_total"].sum(),
            "cash_total": group["cash_total"].sum()
        }
        for b in buckets:
            totals[f"gold_{b}"] = group[f"gold_{b}"].sum()
            totals[f"cash_{b}"] = group[f"cash_{b}"].sum()

        for idx, (_, r) in enumerate(group.iterrows()):
            row_h = 7
            fill = idx % 2 == 1
            pdf.set_fill_color(230, 230, 230) if fill else pdf.set_fill_color(255, 255, 255)

            # Customer name
            customer_name = truncate_text(pdf, reshape_text(r["Customer"]), name_w - 2)
            pdf.cell(name_w, row_h, customer_name, border=1, align="R", fill=fill)

            # Balance
            pdf.cell(sub_w_g21, row_h, format_number(r["total_gold_due"]), border=1, align="C", fill=fill)
            pdf.cell(sub_w_egp, row_h, format_number(r["total_cash_due"]), border=1, align="C", fill=fill)

            # Buckets
            for b in buckets:
                pdf.cell(sub_w_g21, row_h, format_number(r[f"gold_{b}"]), border=1, align="C", fill=fill)
                pdf.cell(sub_w_egp, row_h, format_number(r[f"cash_{b}"]), border=1, align="C", fill=fill)

            # Totals
            pdf.cell(sub_w_g21, row_h, format_number(r["gold_total"]), border=1, align="C", fill=fill)
            pdf.cell(sub_w_egp, row_h, format_number(r["cash_total"]), border=1, align="C", fill=fill)

            pdf.ln(row_h)

        # Total Row
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(name_w, row_h, reshape_text("Total"), border=1, align="C", fill=True)
        pdf.cell(sub_w_g21, row_h, format_number(totals["total_gold_due"]), border=1, align="C", fill=True)
        pdf.cell(sub_w_egp, row_h, format_number(totals["total_cash_due"]), border=1, align="C", fill=True)

        for b in buckets:
            pdf.cell(sub_w_g21, row_h, format_number(totals[f"gold_{b}"]), border=1, align="C", fill=True)
            pdf.cell(sub_w_egp, row_h, format_number(totals[f"cash_{b}"]), border=1, align="C", fill=True)

        pdf.cell(sub_w_g21, row_h, format_number(totals["gold_total"]), border=1, align="C", fill=True)
        pdf.cell(sub_w_egp, row_h, format_number(totals["cash_total"]), border=1, align="C", fill=True)

        pdf.ln(10)
    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output

def build_detailed_pdf(detail_df, summary_df, sp_name, as_of, selected_customer, grace, length):
    """Generate detailed PDF for aging report with alternating row colors, gray headers, and boxed customer sections."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 10)
    
    execution_date = datetime.now().strftime("%d/%m/%Y ")
    pdf.cell(0, 5, reshape_text("New Egypt Gold |تقرير تفصيلي للاعمار الديون "), align="R", ln=0)
    pdf.ln(5)
    pdf.cell(0, 5, f"Execution Date: {execution_date}", align="L", ln=0)
    pdf.ln(10)

    # Parameters Section Header
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 8, reshape_text("Parameters | المعايير"), border=1, align="C", fill=True, ln=1)
    pdf.ln(2)

    params = {
        "Sales Person": sp_name,
        "Customer": selected_customer,
        "Due Date": as_of.strftime('%d/%m/%Y'),
        "Grace Period": f"{grace} يوم",
    }
    draw_parameters_table(pdf, params, 120, [40, 80])
    pdf.ln(10)
    
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 8, reshape_text("Customer Delays By Custom Range"), border=1, align="C", fill=True, ln=1)
    pdf.cell(30, 5, reshape_text("Due Date:"), align="L", ln=0)
    pdf.cell(30, 5, as_of.strftime("%d/%m/%Y"), align="L", ln=1)

    total_overdue_g21 = detail_df["Overdue G21"].sum()
    total_overdue_egp = detail_df["Overdue EGP"].sum()
    
    for customer in sorted(set(summary_df["Customer"])):
        group = detail_df[detail_df["Customer Name"] == customer]
        if not group.empty:
            customer_summary = summary_df[summary_df["Customer"] == customer].iloc[0]
            totals = {
                "cash_due": customer_summary["total_cash_due"],
                "gold_due": customer_summary["total_gold_due"],
                "cash_overdue": customer_summary["cash_total"],
                "gold_overdue": customer_summary["gold_total"]
            }

            # Draw customer header in a box
            box_y = pdf.get_y()
            box_height = 14
            pdf.set_fill_color(230, 230, 230)
            pdf.rect(x=10, y=box_y, w=190, h=box_height)
            pdf.set_y(box_y + 2)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 6, reshape_text(f"العميل: {customer}"), align="C")
            pdf.set_x(10)
            pdf.cell(0, 5, reshape_text(f"اجمالي المديونية الذهب : {format_number(totals['gold_due'])}    |    إجمالي المديونية النقدية: {format_number(totals['cash_due'])}"), align="C", ln=1)
            pdf.ln(2)

            # Table headers
            headers = ["رقم الفاتورة", "تاريخ الفاتورة", "المتأخرة G21", "المتأخرة EGP", "عدد أيام التأخير"]
            widths = [40, 40, 30, 30, 30]
            pdf.set_fill_color(200, 200, 200)
            pdf.set_text_color(0, 0, 0)
            for w, h in zip(widths, headers):
                pdf.cell(w, 8, reshape_text(h), border=1, align="C", ln=0, fill=True)
            pdf.ln()

            # Rows with alternating fill colors
            for idx, (_, row) in enumerate(group.iterrows()):
                fill = idx % 2 == 1
                pdf.set_fill_color(230, 230, 230) if fill else pdf.set_fill_color(255, 255, 255)
                cells = [
                    row["Invoice Ref"],
                    str(row["Invoice Date"]),
                    format_number(row["Overdue G21"]),
                    format_number(row["Overdue EGP"]),
                    str(row["Delay Days"])
                ]
                for w, text in zip(widths, cells):
                    pdf.cell(w, 10, reshape_text(text), border=1, align="C", ln=0, fill=True)
                pdf.ln()
            pdf.ln(4)
            # Total row per customer
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(sum(widths[:2]), 8, reshape_text("Total"), border=1, align="C", fill=True, ln=0)
            pdf.cell(widths[2], 8, format_number(group["Overdue G21"].sum()), border=1, align="C", fill=True, ln=0)
            pdf.cell(widths[3], 8, format_number(group["Overdue EGP"].sum()), border=1, align="C", fill=True, ln=0)
            pdf.cell(widths[4], 8, "", border=1, align="C", fill=True, ln=1)

            pdf.ln(6)

    # Grand total at the end
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(sum(widths[:2]), 8, reshape_text("Grand Total"), border=1, align="C", fill=True, ln=0)
    pdf.cell(widths[2], 8, format_number(total_overdue_g21), border=1, align="C", fill=True, ln=0)
    pdf.cell(widths[3], 8, format_number(total_overdue_egp), border=1, align="C", fill=True, ln=0)
    pdf.cell(widths[4], 8, "", border=1, align="C", fill=True, ln=1)  # Empty cell for Delay Days
    
    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output
# ----------------- Chart Functions -----------------
def setup_arabic_font():
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family='DejaVu Sans')
        return prop
    else:
        st.warning("Arabic font not found. Using default font (may not support Arabic).")
        return None

def create_pie_chart(summary_df, buckets, type="cash"):
    total_overdues = {b: summary_df[f"{type}_{b}"].sum() for b in buckets}
    total = sum(total_overdues.values())
    if total == 0:
        return None

    sizes = [total_overdues[b] for b in buckets]
    outer_labels = [reshape_text(f"الفترة: {b}") for b in buckets]
    prop = setup_arabic_font()
    plt.figure(figsize=(8, 4))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=outer_labels,
        startangle=140,
        labeldistance=1.1,
        pctdistance=0.65,
        autopct=lambda pct: reshape_text(f"{format_number(pct * total / 100)}\n{pct:.1f}%"),
        textprops={'fontproperties': prop, 'fontsize': 8},
    )
    for txt in autotexts:
        txt.set_color('white')
        txt.set_fontproperties(prop)
        txt.set_fontsize(9)
    title = "توزيع التأخيرات حسب الفترة " + ("(كاش)" if type == "cash" else "(ذهب)")
    plt.title(reshape_text(title), fontproperties=prop)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

def create_bar_chart(summary_df, buckets, type="cash"):
    df = summary_df.copy()
    df["total_overdue"] = df[f"{type}_total"]
    top_10 = df.nlargest(10, "total_overdue")
    if top_10["total_overdue"].sum() == 0:
        return None

    customers = top_10["Customer"]
    overdues = top_10["total_overdue"]
    labels = [reshape_text(c) for c in customers]
    prop = setup_arabic_font()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), overdues, tick_label=labels)
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.02 * overdues.max(),
            format_number(y),
            ha='center', va='bottom',
            fontproperties=prop,
            fontsize=9,
        )
    title = f"أعلى 10 عملاء بالمتأخرات ({'كاش' if type == 'cash' else 'ذهب'})"
    plt.title(reshape_text(title), fontproperties=prop, fontsize=11)
    plt.xticks(rotation=45, ha="right", fontproperties=prop, fontsize=9)
    plt.yticks([])
    plt.ylabel("")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, left=0.1)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

# ----------------- Streamlit App -----------------
def aging_report(user_id, log_report):
    engine, err = create_db_engine()
    if err:
        st.error("خطأ في الاتصال: " + err)
        return

    sps = get_salespersons(engine)
    sp_options = ["All"] + sps["name"].tolist()
    sel = st.sidebar.selectbox("Sales Person", sp_options)
    if sel == "---":
        st.info("اختر Sales Person")
        return

    sp_id = None if sel == "All" else (int(sps.loc[sps["name"] == sel, "recordid"].iloc[0]))
    customers = get_customers(engine, sp_id)
    customer_options = ["الكل"] + customers["name"].tolist()
    selected_customer = st.sidebar.selectbox("Customer Name", customer_options)

    as_of = st.sidebar.date_input("Due Date", date.today())
    grace = st.sidebar.number_input("Grace Period", 0, 100, 30)
    length = st.sidebar.number_input("Period Length", 1, 365, 15)
    report_type = st.sidebar.selectbox("Report Type", ["Summary Report", "Details Report"])

    if st.sidebar.button("Generate"):
        summary_df, buckets, detail_df = get_overdues(engine, sp_id, as_of, grace, length)
        if summary_df.empty:
            st.warning("لا توجد متأخرات أو أرصدة لهذا المندوب.")
            return

        if selected_customer != "الكل":
            summary_df = summary_df[summary_df["Customer"] == selected_customer]
            detail_df = detail_df[detail_df["Customer Name"] == selected_customer]

        if summary_df.empty:
            st.warning("لا توجد متأخرات أو أرصدة لهذا العميل.")
            return

        st.subheader(f"المتأخرات حتى {as_of} (بعد فترة السماحية {grace} يوم)")

        overdue_buckets = buckets
        cash_grand_total = sum(summary_df[f"cash_{b}"].sum() for b in overdue_buckets)
        gold_grand_total = sum(summary_df[f"gold_{b}"].sum() for b in overdue_buckets)
        total_cash_due = summary_df["total_cash_due"].sum()
        total_gold_due = summary_df["total_gold_due"].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cash Balance", f"{format_number(total_cash_due)} EGP",
                      delta_color="normal" if total_cash_due > 0 else "inverse")
        with col2:
            st.metric("Total Cash Delays", f"{format_number(cash_grand_total)} EGP")
        with col3:
            st.metric("Total Gold Balance", f"{format_number(total_gold_due)} G21",
                      delta_color="normal" if total_gold_due > 0 else "inverse")
        with col4:
            st.metric("Total Gold Delays", f"{format_number(gold_grand_total)} G21")

        # Display Charts
        st.subheader("تحليل المتأخرات")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**توزيع التأخيرات حسب الفترة (كاش)**")
            pie_chart_cash = create_pie_chart(summary_df, buckets, type="cash")
            if pie_chart_cash:
                st.image(pie_chart_cash)
            st.markdown("**توزيع التأخيرات حسب الفترة (ذهب)**")
            pie_chart_gold = create_pie_chart(summary_df, buckets, type="gold")
            if pie_chart_gold:
                st.image(pie_chart_gold)
        with col2:
            st.markdown("**أعلى 10 عملاء بالمتأخرات (كاش)**")
            bar_chart_cash = create_bar_chart(summary_df, buckets, type="cash")
            if bar_chart_cash:
                st.image(bar_chart_cash)
            st.markdown("**أعلى 10 عملاء بالمتأخرات (ذهب)**")
            bar_chart_gold = create_bar_chart(summary_df, buckets, type="gold")
            if bar_chart_gold:
                st.image(bar_chart_gold)

        if report_type == "Summary Report":
            st.markdown("**المتأخرات**")
            columns = ["Code", "Customer", "total_gold_due", "total_cash_due"]
            for b in buckets:
                columns.append(f"gold_{b}")
                columns.append(f"cash_{b}")
            columns.extend(["gold_total", "cash_total"])

            display_df = summary_df[columns].copy()
            display_df["total_gold_due"] = display_df["total_gold_due"].apply(format_number)
            display_df["total_cash_due"] = display_df["total_cash_due"].apply(format_number)
            display_df["gold_total"] = display_df["gold_total"].apply(format_number)
            display_df["cash_total"] = display_df["cash_total"].apply(format_number)
            for b in buckets:
                display_df[f"gold_{b}"] = display_df[f"gold_{b}"].apply(format_number)
                display_df[f"cash_{b}"] = display_df[f"cash_{b}"].apply(format_number)

            column_mapping = {
                "Code": "Customer Ref",
                "Customer": "Customer Name",
                "total_gold_due": "Total G21 Balance",
                "total_cash_due": "Total EGP Balance",
                "gold_total": "Total G21 Delay",
                "cash_total": "Total EGP Delay"
            }
            for b in buckets:
                display_label = f"من {b.replace('-', ' إلى ').replace('>', 'أكبر من ')} يوم"
                column_mapping[f"gold_{b}"] = f"{display_label} G21"
                column_mapping[f"cash_{b}"] = f"{display_label} EGP"

            def highlight_negatives(s):
                if s.name in ["Total G21 Balance", "Total EGP Balance"]:
                    return ['background-color: red' if v.startswith('(') else '' for v in s]
                return [''] * len(s)

            st.dataframe(
                display_df.rename(columns=column_mapping).style.apply(highlight_negatives, axis=0),
                use_container_width=True
            )

            pdf = build_summary_pdf(summary_df, sel, as_of, buckets, selected_customer, grace, length)
            filename = f"summary_overdues_{sel}_{as_of}.pdf"

        else:
            st.subheader("تفاصيل متأخرات العملاء")
            customers = set(summary_df["Customer"])
            if customers:
                st.markdown("**تفاصيل الفواتير المتأخرة (بعد فترة السماحية)**")
                for customer in sorted(customers):
                    group = detail_df[detail_df["Customer Name"] == customer]
                    if not group.empty:  # Only include customers with overdue invoices
                        customer_summary = summary_df[summary_df["Customer"] == customer]
                        total_cash_due = customer_summary["total_cash_due"].iloc[0] if not customer_summary.empty else 0.0
                        total_gold_due = customer_summary["total_gold_due"].iloc[0] if not customer_summary.empty else 0.0
                        total_cash_overdue = customer_summary["cash_total"].iloc[0] if not customer_summary.empty else 0.0
                        total_gold_overdue = customer_summary["gold_total"].iloc[0] if not customer_summary.empty else 0.0

                        st.markdown(f"**العميل: {customer} (كود: {customer_summary['Code'].iloc[0] if not customer_summary.empty else '-'})**")
                        color_cash = "green" if total_cash_due <= 0 else "red"
                        color_gold = "green" if total_gold_due <= 0 else "blue"
                        st.markdown(f"<span style='color: {color_gold};'>إجمالي المديونية الذهبية: {format_number(total_gold_due)}</span> | "
                                    f"<span style='color: {color_cash};'>إجمالي المديونية النقدية: {format_number(total_cash_due)}</span>",
                                    unsafe_allow_html=True)
                        st.markdown(f"إجمالي المتأخرات الذهبية: {format_number(total_gold_overdue)} | "
                                    f"إجمالي المتأخرات النقدية: {format_number(total_cash_overdue)}",
                                    unsafe_allow_html=True)

                        display_group = group[["Invoice Ref", "Invoice Date", "Overdue G21", "Overdue EGP", "Delay Days"]].copy()
                        display_group["Overdue G21"] = display_group["Overdue G21"].apply(format_number)
                        display_group["Overdue EGP"] = display_group["Overdue EGP"].apply(format_number)
                        st.dataframe(
                            display_group.rename(columns={
                                "Invoice Ref": "Invoice Ref",
                                "Invoice Date": "Invoice Date",
                                "Overdue G21": "G21 Delay",
                                "Overdue EGP": "EGP Delay",
                                "Delay Days": "Delay Days"
                            }),
                            use_container_width=True
                        )
            else:
                st.warning("لا توجد فواتير متأخرة أو أرصدة.")

            pdf = build_detailed_pdf(detail_df, summary_df, sel, as_of, selected_customer, grace, length)
            filename = f"detailed_overdues_{sel}_{as_of}.pdf"

        if pdf and (isinstance(pdf, (bytes, str))) and len(pdf) > 0:
          data = pdf if isinstance(pdf, (bytes, bytearray)) else pdf.encode('latin-1')
          st.download_button("⬇️ تحميل PDF", data, filename, "application/pdf")
        else:
          st.error(...)
 

if __name__ == "__main__":
    main()
