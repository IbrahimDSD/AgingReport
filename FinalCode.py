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
from passlib.hash import pbkdf2_sha256
import matplotlib.pyplot as plt
import os
from io import BytesIO
import matplotlib.font_manager as fm
import sqlitecloud

# SQLite Cloud database connection details
USER_DB_URI = (
    "sqlitecloud://cpran7d0hz.g2.sqlite.cloud:8860/"
    "user_management.db?apikey=oUEez4Dc0TFsVVIVFu8SDRiXea9YVQLOcbzWBsUwZ78"
)

# ----------------- Authentication Setup -----------------
def get_connection():
    try:
        return sqlitecloud.connect(USER_DB_URI)
    except Exception as e:
        st.error(f"خطأ في الاتصال: {e}")
        return None

@st.cache_data(ttl=300)
def get_all_users():
    conn = get_connection()
    if conn:
        df = pd.read_sql("SELECT id, username, role, permissions, full_name FROM users", conn)
        conn.close()
        return df
    return pd.DataFrame(columns=['id', 'username', 'role', 'permissions', 'full_name'])

def get_user_record(username: str):
    conn = get_connection()
    if conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, password_hash, permissions, role FROM users WHERE username = ?",
            (username,)
        )
        rec = c.fetchone()
        conn.close()
        return rec
    return None

def check_login(username: str, password: str) -> bool:
    rec = get_user_record(username)
    if not rec:
        return False
    user_id, pw_hash, permissions, role = rec
    return pbkdf2_sha256.verify(password, pw_hash)

# ----------------- Helper Functions -----------------
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
    credits = [c for c in credits if c["date"] <= as_of]
    pri = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))
    reg = deque(sorted(
        [d for d in debits if d["date"] <= as_of and d["functionid"] not in priority_fids],
        key=lambda x: (x["date"], x["invoiceref"])
    ))

    excess = 0.0
    for cr in sorted(credits, key=lambda x: (x["date"], x.get("invoiceref", ""))):
        rem = cr["amount"]
        while rem > 0 and pri:
            d = pri[0]
            ap = min(rem, d["remaining"])
            d["remaining"] -= ap
            rem -= ap
            if d["remaining"] <= 0:
                d["paid_date"] = cr["date"]
                pri.popleft()
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
        value = round(float(value), 2)
        if value < 0:
            return f"({abs(value):,.2f})"
        elif value == 0:
            return "-"
        else:
            return f"{value:,.2f}"
    except (ValueError, TypeError):
        return str(value)

# ----------------- Data Fetching Functions -----------------
@st.cache_data(ttl=600)
def get_salespersons(_engine):
    return pd.read_sql("SELECT recordid, name FROM sasp ORDER BY name", _engine)

@st.cache_data(ttl=600)
def get_customers(_engine, sp_id):
    if sp_id is None:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid, COALESCE(sasp.name, 'غير محدد') AS sp_name
            FROM fiacc acc
            LEFT JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.groupid = 1
            ORDER BY acc.name
            """
        return pd.read_sql(text(sql), _engine)
    else:
        sql = """
            SELECT DISTINCT acc.recordid, acc.name, acc.spid, sasp.name AS sp_name
            FROM fiacc acc
            JOIN sasp ON acc.spid = sasp.recordid
            WHERE acc.spid = :sp
            ORDER BY acc.name
            """
        return pd.read_sql(text(sql), _engine, params={"sp": sp_id})

@st.cache_data(ttl=300)
def get_overdues(_engine, sp_id, as_of, grace, length):
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

        pc, net_cash = process_fifo(cash_debits, cash_credits, pd.to_datetime(as_of))
        pg, net_gold = process_fifo(gold_debits, gold_credits, pd.to_datetime(as_of))

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
            key=lambda col: col.astype(str),
            inplace=True
        )
    return summary_df, buckets, detail_df

# ----------------- PDF Generation Functions -----------------
def truncate_text(pdf, text, width):
    ellipsis = "..."
    while pdf.get_string_width(ellipsis + text) > width and len(text) > 0:
        text = text[1:]
    if pdf.get_string_width(ellipsis + text) <= width:
        text = ellipsis + text
    return text


def draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w):
    pdf.cell(name_w, 8, reshape_text("Name"), border=1, align="C", ln=0)
    pdf.cell(bal_w, 8, reshape_text("Balance"), border=1, align="C", ln=0)
    for b in buckets:
        pdf.cell(bucket_w, 8, reshape_text(f"From {b.replace('-', ' - ')}"), border=1, align="C", ln=0)
    pdf.cell(tot_w, 8, reshape_text("Total Delay"), border=1, align="C", ln=1)
    pdf.cell(name_w, 8, "", border=1, ln=0)
    pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=0)
    for _ in buckets:
        pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
        pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "G21", border=1, align="C", ln=0)
    pdf.cell(sub_w, 8, "EGP", border=1, align="C", ln=1)


def draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths):
    parameters = [
        ("المندوب", sp_name),
        ("العميل", selected_customer),
        ("تاريخ الاستحقاق", as_of.strftime('%d/%m/%Y')),
        ("فترة السماحية", f"{grace} يوم"),
        ("مدة الفترة", f"{length} يوم")
    ]
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(col_widths[0], 8, reshape_text("المعامل"), border=1, align="C", fill=True, ln=0)
    pdf.cell(col_widths[1], 8, reshape_text("القيمة"), border=1, align="C", fill=True, ln=1)
    for label, value in parameters:
        pdf.cell(col_widths[0], 8, reshape_text(label), border=1, align="R", ln=0)
        pdf.cell(col_widths[1], 8, reshape_text(value), border=1, align="R", ln=1)


def build_summary_pdf(df, sp_name, as_of, buckets, selected_customer, grace, length):
    pdf = FPDF(orientation="L", unit="mm", format="A3")
    pdf.add_page()
    pdf.add_font('DejaVu', '','DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)

    exe = datetime.now().strftime("%d/%m/%Y %I:%M %p")
    pdf.set_xy(10, 10)
    pdf.cell(0, 5, reshape_text("New Egypt Gold | تقرير متأخرات"), ln=0, align="c")
    pdf.ln(5)
    pdf.cell(0, 5, f"Execution Date: {exe}", ln=0, align="L")
    pdf.ln(10)

    table_width = 120
    col_widths = [40, 80]
    pdf.set_xy(10, pdf.get_y())
    draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths)
    pdf.ln(10)

    name_w = 50
    bal_w = 60
    bucket_w = 60
    tot_w = 60
    sub_w = bal_w / 2
    line_h = 7
    bottom_margin = 20

    if sp_name == "All":
        grouped = df.groupby("sp_id")
    else:
        grouped = [(sp_name, df)]

    for sp_id, group in grouped:
        sp_display_name = group["sp_name"].iloc[0] if sp_name == "All" else sp_name
        pdf.set_xy(10, pdf.get_y())
        pdf.cell(0, 5, reshape_text(f"Sales Person: {sp_display_name}"), border=0, ln=1, align="L")
        pdf.ln(4)
        draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w)

        for _, r in group.iterrows():
            row_h = line_h  # Start with minimum height
            # Calculate height for total_gold_due
            lines_g21 = pdf.multi_cell(sub_w, line_h, format_number(r["total_gold_due"]), border=0, align="R",
                                       split_only=True)
            g21_h = len(lines_g21) * line_h
            row_h = max(row_h, g21_h)
            # Calculate height for total_cash_due
            lines_egp = pdf.multi_cell(sub_w, line_h, format_number(r["total_cash_due"]), border=0, align="R",
                                       split_only=True)
            egp_h = len(lines_egp) * line_h
            row_h = max(row_h, egp_h)
            # Calculate heights for each bucket
            for b in buckets:
                lines_gold = pdf.multi_cell(sub_w, line_h, format_number(r[f"gold_{b}"]), border=0, align="R",
                                            split_only=True)
                gold_h = len(lines_gold) * line_h
                lines_cash = pdf.multi_cell(sub_w, line_h, format_number(r[f"cash_{b}"]), border=0, align="R",
                                            split_only=True)
                cash_h = len(lines_cash) * line_h
                row_h = max(row_h, gold_h, cash_h)
            # Calculate heights for totals
            lines_tot_g21 = pdf.multi_cell(sub_w, line_h, format_number(r["gold_total"]), border=0, align="R",
                                           split_only=True)
            tot_g21_h = len(lines_tot_g21) * line_h
            lines_tot_egp = pdf.multi_cell(sub_w, line_h, format_number(r["cash_total"]), border=0, align="R",
                                           split_only=True)
            tot_egp_h = len(lines_tot_egp) * line_h
            row_h = max(row_h, tot_g21_h, tot_egp_h)

            # Check for page break
            if pdf.get_y() + row_h + bottom_margin > pdf.h:
                pdf.add_page()
                pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
                pdf.set_font('DejaVu', '', 12)
                pdf.cell(0, 5, reshape_text(f"Sales Person: {sp_display_name}"), border=0, ln=1, align="L")
                pdf.ln(4)
                draw_table_headers(pdf, buckets, name_w, bal_w, bucket_w, tot_w, sub_w)

            x0, y0 = pdf.get_x(), pdf.get_y()

            # Draw Customer Name
            customer_name = reshape_text(r["Customer"])
            if pdf.get_string_width(customer_name) > name_w - 2:
                customer_name = truncate_text(pdf, customer_name, name_w - 2)
            pdf.cell(name_w, line_h, customer_name, border=1, align="L", ln=0)

            # Draw Total Gold Due
            pdf.set_xy(x0 + name_w, y0)
            color = (0, 128, 0) if r["total_gold_due"] <= 0 else (0, 0, 255)
            pdf.set_text_color(*color)
            pdf.multi_cell(sub_w, line_h, format_number(r["total_gold_due"]), border=1, align="R")
            pdf.set_text_color(0, 0, 0)

            # Draw Total Cash Due
            pdf.set_xy(x0 + name_w + sub_w, y0)
            color = (0, 128, 0) if r["total_cash_due"] <= 0 else (255, 0, 0)
            pdf.set_text_color(*color)
            pdf.multi_cell(sub_w, line_h, format_number(r["total_cash_due"]), border=1, align="R")
            pdf.set_text_color(0, 0, 0)

            # Draw Buckets
            x_b = x0 + name_w + bal_w
            for i, b in enumerate(buckets):
                pdf.set_xy(x_b + i * bucket_w, y0)
                pdf.multi_cell(sub_w, line_h, format_number(r[f"gold_{b}"]), border=1, align="R")
                pdf.set_xy(x_b + i * bucket_w + sub_w, y0)
                pdf.multi_cell(sub_w, line_h, format_number(r[f"cash_{b}"]), border=1, align="R")

            # Draw Totals
            x_t = x_b + len(buckets) * bucket_w
            pdf.set_xy(x_t, y0)
            pdf.multi_cell(sub_w, line_h, format_number(r["gold_total"]), border=1, align="R")
            pdf.set_xy(x_t + sub_w, y0)
            pdf.multi_cell(sub_w, line_h, format_number(r["cash_total"]), border=1, align="R")

            # Move to the next row position
            pdf.set_xy(x0, y0 + row_h)

        pdf.ln(10)

    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, bytearray) else out


def build_detailed_pdf(detail_df, summary_df, sp_name, as_of, selected_customer, grace, length):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)

    execution_date = datetime.now().strftime("%d/%m/%Y %H:%M %p")
    pdf.set_xy(10, 10)
    pdf.cell(0, 5, reshape_text(f"New Egypt Gold | تقرير تفصيلي للمتأخرات"), border=0, ln=0, align="R")
    pdf.cell(-50, 5, f"ITS-08223 / EGS", border=0, ln=0, align="R")
    pdf.ln(5)
    pdf.cell(0, 5, f"Execution Date: {execution_date}", border=0, ln=0, align="L")
    pdf.cell(-50, 5, f"Page Number: 1/1", border=0, ln=0, align="R")
    pdf.ln(10)

    table_width = 120
    col_widths = [40, 80]
    pdf.set_xy(10, pdf.get_y())
    draw_parameters_table(pdf, sp_name, selected_customer, as_of, grace, length, table_width, col_widths)
    pdf.ln(10)

    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 8, reshape_text("Customer Delays By Custom Range."), border=1, ln=1, align="C", fill=True)
    pdf.cell(30, 5, reshape_text("Due Date:"), border=0, ln=0, align="L")
    pdf.cell(30, 5, as_of.strftime("%d/%m/%Y"), border=0, ln=0, align="L")
    pdf.ln(5)

    customers = set(summary_df["Customer"])
    for customer in sorted(customers):
        group = detail_df[detail_df["Customer Name"] == customer]
        if not group.empty:  # Only include customers with overdue invoices
            customer_summary = summary_df[summary_df["Customer"] == customer]
            total_cash_due = customer_summary["total_cash_due"].iloc[0] if not customer_summary.empty else 0.0
            total_gold_due = customer_summary["total_gold_due"].iloc[0] if not customer_summary.empty else 0.0
            total_cash_overdue = customer_summary["cash_total"].iloc[0] if not customer_summary.empty else 0.0
            total_gold_overdue = customer_summary["gold_total"].iloc[0] if not customer_summary.empty else 0.0

            pdf.set_xy(10, pdf.get_y())
            pdf.multi_cell(0, 5, reshape_text(f"العميل: {customer}"), border=0, align="R")
            pdf.set_xy(10, pdf.get_y())
            pdf.set_text_color(0, 128, 0) if total_cash_due <= 0 else pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 5, reshape_text(f"إجمالي المديونية النقدية: {format_number(total_cash_due)}"), border=0,
                     ln=1, align="R")
            pdf.set_text_color(0, 128, 0) if total_gold_due <= 0 else pdf.set_text_color(0, 0, 255)
            pdf.cell(0, 5, reshape_text(f"إجمالي المديونية الذهبية: {format_number(total_gold_due)}"), border=0,
                     ln=1, align="R")
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, reshape_text(f"إجمالي المتأخرات النقدية: {format_number(total_cash_overdue)}"), border=0,
                     ln=1, align="R")
            pdf.cell(0, 5, reshape_text(f"إجمالي المتأخرات الذهبية: {format_number(total_gold_overdue)}"), border=0,
                     ln=1, align="R")
            pdf.ln(4)

            headers = ["رقم الفاتورة", "تاريخ الفاتورة", "المتأخرة G21", "المتأخرة EGP", "عدد أيام التأخير"]
            widths = [40, 40, 30, 30, 30]
            for w, h in zip(widths, headers):
                pdf.cell(w, 8, reshape_text(h), border=1, ln=0, align="C")
            pdf.ln()
            for _, row in group.iterrows():
                pdf.cell(40, 10, reshape_text(row["Invoice Ref"]), border=1, align="C", ln=0)
                pdf.cell(40, 10, str(row["Invoice Date"]), border=1, align="C", ln=0)
                pdf.cell(30, 10, format_number(row["Overdue G21"]), border=1, align="R", ln=0)
                pdf.cell(30, 10, format_number(row["Overdue EGP"]), border=1, align="R", ln=0)
                pdf.cell(30, 10, str(row["Delay Days"]), border=1, align="R", ln=1)
            pdf.ln(4)

    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output

# ----------------- Chart Generation Functions -----------------
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

# ----------------- Streamlit Application -----------------
def main():
    st.set_page_config(page_title="Aging Report", layout="wide")
    st.title("📊 تقرير متأخرات العملاء حسب Sales Person")

    # Check login state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.subheader("تسجيل الدخول")
        username = st.text_input("اسم المستخدم")
        password = st.text_input("كلمة المرور", type="password")
        if st.button("تسجيل الدخول"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("تم تسجيل الدخول بنجاح!")
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")
    else:
        # Place logout button at the top of the sidebar
        with st.sidebar:
            if st.button("تسجيل الخروج"):
                st.session_state.logged_in = False
                st.rerun()
            st.markdown("---")  # Separator for better UI

        engine, err = create_db_engine()
        if err:
            st.error("خطأ في الاتصال: " + err)
            return

        sps = get_salespersons(engine)
        sp_options = ["All"] + sps["name"].tolist()
        with st.sidebar:
            sel = st.selectbox("Sales Person", sp_options)
        if sel == "---":
            st.info("اختر Sales Person")
            return

        sp_id = None if sel == "All" else (int(sps.loc[sps["name"] == sel, "recordid"].iloc[0]))
        customers = get_customers(engine, sp_id)
        customer_options = ["الكل"] + customers["name"].tolist()
        with st.sidebar:
            selected_customer = st.selectbox("Customer Name", customer_options)
            as_of = st.date_input("Due Date", date.today())
            grace = st.number_input("Grace Period", 0, 100, 30)
            length = st.number_input("Period Length", 1, 365, 15)
            report_type = st.selectbox("Report Type", ["Summary Report", "Details Report"])
            generate_button = st.button("Generate")

        if generate_button:
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
                        total_cash_due = customer_summary["total_cash_due"].iloc[
                            0] if not customer_summary.empty else 0.0
                        total_gold_due = customer_summary["total_gold_due"].iloc[
                            0] if not customer_summary.empty else 0.0
                        total_cash_overdue = customer_summary["cash_total"].iloc[
                            0] if not customer_summary.empty else 0.0
                        total_gold_overdue = customer_summary["gold_total"].iloc[
                            0] if not customer_summary.empty else 0.0

                        st.markdown(
                            f"**العميل: {customer} (كود: {customer_summary['Code'].iloc[0] if not customer_summary.empty else '-'})**")
                        color_cash = "green" if total_cash_due <= 0 else "red"
                        color_gold = "green" if total_gold_due <= 0 else "blue"
                        st.markdown(
                            f"<span style='color: {color_gold};'>إجمالي المديونية الذهبية: {format_number(total_gold_due)}</span> | "
                            f"<span style='color: {color_cash};'>إجمالي المديونية النقدية: {format_number(total_cash_due)}</span>",
                            unsafe_allow_html=True)
                        st.markdown(f"إجمالي المتأخرات الذهبية: {format_number(total_gold_overdue)} | "
                                    f"إجمالي المتأخرات النقدية: {format_number(total_cash_overdue)}",
                                    unsafe_allow_html=True)

                        display_group = group[
                            ["Invoice Ref", "Invoice Date", "Overdue G21", "Overdue EGP", "Delay Days"]].copy()
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
