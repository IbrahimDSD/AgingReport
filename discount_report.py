import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from collections import deque
from datetime import datetime, date
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------- Constants --------------
DB_CONFIG = {
    "server": "52.48.117.197",
    "database": "R1029",
    "username": "sa",
    "password": "Argus@NEG",
    "driver": "ODBC Driver 17 for SQL Server"
}

# -------------- Helpers --------------
def reshape_text(txt):
    """Reshape text for Arabic display."""
    if not isinstance(txt, str):
        txt = str(txt)
    try:
        return get_display(arabic_reshaper.reshape(txt))
    except Exception as e:
        logger.error(f"Error reshaping text: {e}, text={txt}")
        return str(txt)

def format_number(value):
    """Format numeric values for display."""
    try:
        value = float(value)
        return f"({abs(value):,.2f})" if value < 0 else "-" if value == 0 else f"{value:,.2f}"
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting number: {e}, value={value}")
        return str(value)

# -------------- Database Setup --------------
def create_db_engine():
    """Create SQLAlchemy engine with connection pooling."""
    try:
        connection_string = (
            f"DRIVER={DB_CONFIG['driver']};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
            f"TrustServerCertificate=Yes;"
            f"Connection Timeout=30"
        )
        encoded_connection = quote_plus(connection_string)
        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={encoded_connection}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except Exception as e:
        return None, str(e)

# -------------- Discount Report Functions --------------
@st.cache_data(ttl=600, hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def fetch_data(query, params=None):
    """Fetch data from database."""
    engine, error = create_db_engine()
    if error:
        st.error(f"❌ Database connection failed: {error}")
        logger.error(f"Database connection failed: {error}")
        return None
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
            logger.info(f"Fetched {len(df)} rows from query: {query[:100]}...")
            return df
    except SQLAlchemyError as e:
        st.error(f"❌ Error fetching data: {e}")
        logger.error(f"Error fetching data: {e}")
        return None

def calculate_vat(row):
    """Calculate VAT based on currency type."""
    try:
        return row['amount'] * (11.18 if row['currencyid'] == 2 else 7.45 if row['currencyid'] == 3 else 0.0)
    except (TypeError, ValueError) as e:
        logger.error(f"Error calculating VAT: {e}, row={row}")
        return 0.0

def convert_gold(row):
    """Convert transaction amounts to standard gold units."""
    try:
        qty = row.get('qty', row['amount'] if pd.isna(row.get('qty')) else np.nan)
        conversions = {
            2: 6/7,
            3: 1.0,
            14: 14/21,
            4: 24/21
        }
        factor = conversions.get(int(row['currencyid']), 1.0)
        return qty * factor if row['reference'].startswith('S') else row['amount'] * factor
    except (TypeError, ValueError) as e:
        logger.error(f"Error in convert_gold: {e}, row={row}")
        return row['amount']

def process_fifo(debits, credits):
    """Process transactions using FIFO for discount report."""
    debits_q = deque(sorted(debits, key=lambda x: x['date']))
    history = []
    for credit in sorted(credits, key=lambda x: x['date']):
        rem = credit['amount']
        while rem > 0 and debits_q:
            d = debits_q[0]
            apply_amt = min(rem, d['remaining'])
            d['remaining'] -= apply_amt
            rem -= apply_amt
            if d['remaining'] <= 0:
                d['paid_date'] = credit['date']
                history.append(debits_q.popleft())
    history.extend(debits_q)
    logger.info(f"Processed FIFO: {len(history)} transactions remaining")
    return history

def process_report(df, currency_type):
    """Process report data for display."""
    if df.empty:
        logger.warning("Empty DataFrame in process_report")
        return df
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['paid_date'] = df['paid_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'Unpaid')
    df['aging_days'] = df.apply(
        lambda r: '-' if r['paid_date'] == 'Unpaid' else str((pd.to_datetime(r['paid_date']) - pd.to_datetime(r['date'])).days),
        axis=1
    )
    for col in ['amount', 'remaining', 'vat_amount']:
        df[col] = df[col].round(2)
    suffix = '_gold' if currency_type != 1 else '_cash'
    return df.add_suffix(suffix).rename(columns={f'date{suffix}': 'date', f'reference{suffix}': 'reference'})

def process_transactions(raw, discounts, extras, start_date):
    """Process raw transactions for discount report."""
    if raw.empty:
        logger.warning("No transactions to process in process_transactions")
        return pd.DataFrame()
    
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw.dropna(subset=['date'])
    logger.info(f"Processing {len(raw)} transactions after date cleaning")
    
    def calc_row(r):
        try:
            base = r['baseAmount'] + r['basevatamount']
            if pd.to_datetime(r['date']) >= start_date:
                disc = discounts.get(r['categoryid'], 0)
                extra = extras.get(r['categoryid'], 0)
                return base - (disc * r['qty']) - (extra * r['qty'])
            return base
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating row: {e}, row={r}")
            return r['baseAmount'] + r['basevatamount']
    
    grouped = raw.groupby(['functionid', 'recordid', 'date', 'reference', 'currencyid', 'amount']).apply(
        lambda g: pd.Series({
            'date': g.iloc[0]['date'],
            'reference': g.iloc[0]['reference'],
            'currencyid': g.iloc[0]['currencyid'],
            'amount': g.loc[~g['baseAmount'].isna(), :].apply(calc_row, axis=1).sum() if g.iloc[0]['reference'].startswith('S') and g.iloc[0]['currencyid'] == 1 else g.iloc[0]['amount'],
            'original_amount': g.iloc[0]['amount']
        })
    ).reset_index(drop=True)
    
    grouped['converted'] = grouped.apply(convert_gold, axis=1)
    logger.info(f"Processed {len(grouped)} transactions after grouping")
    return grouped

def calculate_aging_reports(transactions):
    """Calculate aging reports for discount transactions."""
    if transactions.empty:
        logger.warning("No transactions in calculate_aging_reports")
        return pd.DataFrame()
    
    cash_debits, cash_credits, gold_debits, gold_credits = [], [], [], []
    transactions['vat_amount'] = transactions.apply(calculate_vat, axis=1)
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    
    for _, r in transactions.iterrows():
        try:
            entry = {
                'date': r['date'],
                'reference': r['reference'],
                'amount': abs(r['converted']),
                'remaining': abs(r['converted']),
                'paid_date': None,
                'vat_amount': r['vat_amount']
            }
            if r['currencyid'] == 1:
                (cash_debits if r['amount'] > 0 else cash_credits).append(entry)
            else:
                (gold_debits if r['amount'] > 0 else gold_credits).append(entry)
        except Exception as e:
            logger.error(f"Error processing transaction: {e}, row={r}")
            continue
    
    cash = process_fifo(cash_debits, cash_credits)
    gold = process_fifo(gold_debits, gold_credits)
    cash_df = process_report(pd.DataFrame(cash), 1)
    gold_df = process_report(pd.DataFrame(gold), 2)
    
    df = pd.merge(cash_df, gold_df, on=['date', 'reference'], how='outer').fillna({
        f'{k}_{s}': v for k in ['amount', 'remaining', 'vat_amount'] for s in ['gold', 'cash'] for v in [0]
    }).fillna({
        f'paid_date_{s}': '-' for s in ['gold', 'cash']
    }).fillna({
        f'aging_days_{s}': '-' for s in ['gold', 'cash']
    })
    
    logger.info(f"Generated aging report with {len(df)} rows")
    return df[['date', 'reference', 'amount_gold', 'remaining_gold', 'paid_date_gold', 'aging_days_gold',
              'amount_cash', 'remaining_cash', 'paid_date_cash', 'aging_days_cash']]

def export_pdf(report_df, params):
    """Export discount report to PDF."""
    pdf = FPDF(orientation='L')
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.set_margins(15, 10, 15)
    
    pdf.cell(0, 15, reshape_text("تقرير الخصومات"), align='C', ln=1)
    customer = params.pop("اسم العميل", None)
    if customer:
        pdf.cell(0, 10, f"{reshape_text('اسم العميل')}: {reshape_text(customer)}", align='L', ln=1)
    
    col_width = pdf.w / 2 - 20
    params_list = list(params.items())
    half = len(params_list) // 2
    for i in range(max(half, len(params_list) - half)):
        pdf.set_x(15)
        if i < len(params_list[:half]):
            k, v = params_list[i]
            pdf.cell(col_width, 8, f"{reshape_text(k)}: {reshape_text(v)}", align='L', ln=0)
        if i < len(params_list[half:]):
            k, v = params_list[half + i]
            pdf.cell(col_width, 8, f"{reshape_text(k)}: {reshape_text(v)}", align='L', ln=1)
        else:
            pdf.ln()
    
    if report_df.empty:
        pdf.set_font('DejaVu', '', 14)
        pdf.cell(0, 10, reshape_text("لا توجد بيانات للعرض"), align='C', ln=1)
        pdf.ln(8)
        pdf.set_font('DejaVu', '', 18)
        pdf.cell(0, 8, "Generated by BI", align='R', ln=1)
        logger.info("Generated PDF with no data message")
        return pdf.output(dest='S').encode('latin1') if isinstance(pdf.output(dest='S'), str) else pdf.output(dest='S')
    
    logger.info(f"Rendering PDF with {len(report_df)} rows: {report_df.head().to_dict()}")
    headers = [
        "التاريخ", "الرقم المرجعي", "ذهب عيار 21", "تاريخ سداد الذهب",
        "المبلغ النقدي", "تاريخ سداد النقدية", "أيام سداد الذهب", "أيام سداد النقدية"
    ]
    col_widths = [30, 40, 30, 35, 30, 35, 32, 30]
    pdf.set_fill_color(200, 220, 255)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 10, reshape_text(h), border=1, fill=True, align='C', ln=0)
    pdf.ln()
    
    threshold = params.get("فترة سداد العميل", 0)
    for idx, row in report_df.iterrows():
        try:
            cash_age = int(row['aging_days_cash']) if row['aging_days_cash'] != '-' else 0
            gold_age = int(row['aging_days_gold']) if row['aging_days_gold'] != '-' else 0
            cells = [
                str(row['date']),
                str(row['reference']),
                format_number(float(str(row['amount_gold']).replace(',', '').replace('(', '').replace(')', ''))),
                str(row['paid_date_gold']),
                format_number(float(str(row['amount_cash']).replace(',', '').replace('(', '').replace(')', ''))),
                str(row['paid_date_cash']),
                str(row['aging_days_gold']),
                str(row['aging_days_cash'])
            ]
            fills = [False] * 6 + [gold_age > threshold, cash_age > threshold]
            for w, text, fill in zip(col_widths, cells, fills):
                pdf.cell(w, 7, reshape_text(text), border=1, fill=fill, align='C')
            pdf.ln()
            logger.debug(f"Added row {idx} to PDF: {cells}")
        except Exception as e:
            logger.error(f"Error adding row {idx} to PDF: {e}, row={row.to_dict()}")
            continue
    
    pdf.ln(8)
    pdf.set_font('DejaVu', '', 18)
    pdf.cell(0, 8, "Generated by BI", align='R', ln=1)
    logger.info(f"Generated PDF with {len(report_df)} rows")
    return pdf.output(dest='S').encode('latin1') if isinstance(pdf.output(dest='S'), str) else pdf.output(dest='S')

def discount_report(user_id):
    """Generate discount and payment period report."""
    aging_threshold = st.sidebar.number_input("Enter Aging Days Threshold", min_value=0, value=30, step=1)
    customers = fetch_data("SELECT recordid, name, reference FROM fiacc WHERE groupid = 1")
    if customers is None or customers.empty:
        st.error("لا يمكن جلب قائمة العملاء.")
        logger.error("No customers fetched")
        return
    
    cust_list = ["Select Customer..."] + [f"{r['name']} ({r['reference']})" for _, r in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer Name", cust_list)
    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    # Validate date range
    if start_date > end_date:
        st.error("تاريخ البدء يجب أن يكون قبل تاريخ الانتهاء.")
        logger.error("Invalid date range")
        return
    
    discounts = {
        50: st.sidebar.number_input("احجار عيار 21", 0.0, 1000.0, 0.0),
        61: st.sidebar.number_input("سادة عيار 21", 0.0, 1000.0, 0.0),
        47: st.sidebar.number_input("ذهب مشغول عيار 18", 0.0, 1000.0, 0.0),
        62: st.sidebar.number_input("سادة عيار 18", 0.0, 1000.0, 0.0),
        48: st.sidebar.number_input("ستار 18", 0.0, 1000.0, 0.0)
    }
    extras = {
        50: st.sidebar.number_input("تعجيل دفع عيار 21", 0.0, 1000.0, 0.0),
        61: st.sidebar.number_input("تعجيل دفع عيار 21", 0.0, 1000.0, 0.0, key="extra_61"),
        47: st.sidebar.number_input("تعجيل دفع عيار 18", 0.0, 1000.0, 0.0),
        62: st.sidebar.number_input("تعجيل دفع عيار 18", 0.0, 1000.0, 0.0, key="extra_62")
    }
    
    raw = None
    if selected_customer != "Select Customer...":
        cid = customers.iloc[cust_list.index(selected_customer) - 1]['recordid']
        query = """
            SELECT f.plantid, f.functionid, f.recordid, f.date, f.reference,f.description,
                   f.currencyid, f.amount, s.qty, s.baseAmount, s.basevatamount, ivit.categoryid
            FROM fitrx f
            LEFT JOIN satrx s ON f.functionid=s.functionid AND f.recordid=s.recordid
            LEFT JOIN ivit ON s.itemid=ivit.recordid
            WHERE f.accountid = :acc
        """
        raw = fetch_data(query, {"acc": cid})
        if raw is None or raw.empty:
            st.warning("لا توجد معاملات متاحة للعميل المحدد.")
            logger.warning(f"No transactions for customer ID {cid}")
            return
    
    st.markdown("### اختر العمليات خزينة الخصومات:")
    labels = []
    if raw is not None and not raw.empty:
        tmp = raw[raw['plantid'] == 56].copy()
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        tmp = tmp[tmp['date'] >= pd.to_datetime(start_date)]
        labels = [
            f"{r['functionid']}|{r['recordid']}|{r['date'].date()}|{r['amount']}|{r['reference']}|{r['description']}"
            for _, r in tmp.iterrows()
        ]
    
    overrides = st.multiselect(
        "",
        options=labels,
        format_func=lambda x: f"Reference: {x.split('|')[4]} – Date: {x.split('|')[2]} – Amount: {x.split('|')[3]} - Description: {x.split('|')[5]}"
    )
    
    if not st.sidebar.button("Generate Report"):
        return
    
    start_time = time.time()
    if selected_customer == "Select Customer...":
        st.error("الرجاء اختيار عميل.")
        logger.error("No customer selected")
        return
    if raw is None or raw.empty:
        st.warning("لا توجد معاملات للعميل المحدد.")
        logger.warning("No transactions available after fetch")
        return
    
    raw2 = raw.copy()
    raw2['date'] = pd.to_datetime(raw2['date'], errors='coerce')
    raw2 = raw2.dropna(subset=['date'])
    raw2 = raw2.loc[~((raw2['plantid'] == 56) & (raw2['amount'] > 0))]
    logger.info(f"Filtered to {len(raw2)} transactions after initial processing")
    
    for label in overrides:
        try:
            fid, rid, *_ = label.split('|')
            raw2.loc[(raw2['functionid'] == int(fid)) & (raw2['recordid'] == int(rid)), 'date'] = pd.to_datetime(start_date)
        except Exception as e:
            logger.error(f"Error applying override: {e}, label={label}")
            continue
    
    txs = process_transactions(raw2, discounts, extras, pd.to_datetime(start_date))
    if txs.empty:
        st.warning("لا توجد معاملات بعد المعالجة.")
        logger.warning("No transactions after processing")
        return
    
    report = calculate_aging_reports(txs)
    if report.empty:
        st.warning("لا توجد بيانات للعرض بعد حساب فترات السداد.")
        logger.warning("No data after aging calculations")
        return
    
    # Apply date filtering
    report['date_dt'] = pd.to_datetime(report['date'], errors='coerce')
    report = report[
        (report['date_dt'] >= pd.to_datetime(start_date)) & 
        (report['date_dt'] <= pd.to_datetime(end_date))
    ].sort_values(['date_dt', 'paid_date_cash', 'paid_date_gold']).reset_index(drop=True).drop(columns=['date_dt'])
    logger.info(f"Filtered to {len(report)} transactions after date range")
    
    if report.empty:
        st.warning(f"لا توجد معاملات في الفترة من {start_date} إلى {end_date}. جرب توسيع نطاق التاريخ.")
        logger.warning(f"No transactions in date range {start_date} to {end_date}")
    
    # Ensure numeric columns are properly formatted
    for col in ['amount_cash', 'remaining_cash', 'amount_gold', 'remaining_gold']:
        report[col] = report[col].apply(lambda x: format_number(float(x)) if pd.notna(x) and x != '-' else x)
    
    def highlight_row(row):
        styles = [''] * len(row)
        cash_age = int(row['aging_days_cash']) if row['aging_days_cash'] != '-' else 0
        gold_age = int(row['aging_days_gold']) if row['aging_days_gold'] != '-' else 0
        if cash_age > aging_threshold and gold_age > aging_threshold:
            return ['background-color: #FFCCCB'] * len(row)
        if cash_age > aging_threshold:
            styles[row.index.get_loc('aging_days_cash')] = 'background-color: #FFCCCB'
        if gold_age > aging_threshold:
            styles[row.index.get_loc('aging_days_gold')] = 'background-color: #FFCCCB'
        return styles
    
    st.subheader("تقرير الخصومات وفترات السداد")
    if not report.empty:
        # Debug: Display sample of report_df
        st.write("**معاينة البيانات قبل إنشاء PDF:**")
        st.dataframe(report.head())
        
        display_df = report.rename(columns={
            'date': 'التاريخ',
            'reference': 'الرقم المرجعي',
            'amount_gold': 'ذهب عيار 21',
            'remaining_gold': 'ذهب متبقي',
            'paid_date_gold': 'تاريخ سداد الذهب',
            'aging_days_gold': 'أيام سداد الذهب',
            'amount_cash': 'المبلغ النقدي',
            'remaining_cash': 'نقدي متبقي',
            'paid_date_cash': 'تاريخ سداد النقدية',
            'aging_days_cash': 'أيام سداد النقدية'
        })
        st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)
        
        params = {
            "فترة سداد العميل": aging_threshold,
            "تاريخ البدء": start_date.strftime("%Y-%m-%d"),
            "تاريخ الانتهاء": end_date.strftime("%Y-%m-%d"),
            "احجار عيار 21": f"{discounts[50]:.2f}",
            "سادة عيار 21": f"{discounts[61]:.2f}",
            "ذهب مشغول عيار 18": f"{discounts[47]:.2f}",
            "سادة عيار 18": f"{discounts[62]:.2f}",
            "ستار 18": f"{discounts[48]:.2f}",
            "تعجيل دفع عيار 21": f"{extras[50]:.2f}",
            "تعجيل دفع عيار 18": f"{extras[47]:.2f}"
        }
        if selected_customer:
            params["اسم العميل"] = selected_customer
        
        pdf = export_pdf(report, params)
        if pdf and len(pdf) > 0:
            st.download_button(
                "⬇️ تحميل PDF",
                pdf,
                f"discount_report_{selected_customer}_{start_date}.pdf",
                "application/pdf"
            )
        else:
            st.error("فشل في إنشاء ملف PDF.")
            logger.error("Failed to generate PDF")
    else:
        st.warning("لا توجد بيانات للعرض بعد تطبيق المرشحات. حاول تغيير العميل أو نطاق التاريخ.")
    
    log_report(user_id)
