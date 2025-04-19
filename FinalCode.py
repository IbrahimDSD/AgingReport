import streamlit as st
import pandas as pd
import numpy as np
import sqlitecloud
from datetime import datetime
from passlib.hash import pbkdf2_sha256
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from collections import deque
from urllib.parse import quote_plus
import time
from fpdf import FPDF

# -------------- Helpers --------------
def rerun():
    st.experimental_rerun()

# --------- User‑DB Setup  ---------
USER_DB_URI = (
    "sqlitecloud://cpran7d0hz.g2.sqlite.cloud:8860/"
    "user_management.db?apikey=oUEez4Dc0TFsVVIVFu8SDRiXea9YVQLOcbzWBsUwZ78"
)

def get_sqlitecloud_connection():
    try:
        return sqlitecloud.connect(USER_DB_URI)
    except Exception as e:
        st.error(f"❌ Failed to connect to User DB: {e}")
        return None

def init_user_db():
    conn = get_sqlitecloud_connection()
    if not conn:
        return
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE,
          password_hash TEXT,
          role TEXT,
          full_name TEXT,
          force_password_change INTEGER DEFAULT 0,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS report_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          duration REAL,
          FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("SELECT 1 FROM users WHERE username='admin'")
    if not c.fetchone():
        pw = pbkdf2_sha256.hash("admin123")
        c.execute(
            "INSERT INTO users(username,password_hash,role,full_name,force_password_change) "
            "VALUES (?,?,?,?,1)",
            ("admin", pw, "admin", "System Admin")
        )
    conn.commit()
    conn.close()

init_user_db()

# --------- User Management  ----------
def create_user(username, password, role, full_name):
    conn = get_sqlitecloud_connection()
    if not conn:
        return False
    c = conn.cursor()
    try:
        pw = pbkdf2_sha256.hash(password)
        c.execute(
            "INSERT INTO users(username,password_hash,role,full_name,force_password_change) "
            "VALUES (?,?,?,?,1)",
            (username, pw, role, full_name)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def reset_user_password(user_id, new_password):
    conn = get_sqlitecloud_connection()
    if not conn:
        return
    c = conn.cursor()
    pw = pbkdf2_sha256.hash(new_password)
    c.execute(
        "UPDATE users SET password_hash=?, force_password_change=1 WHERE id=?",
        (pw, user_id)
    )
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = get_sqlitecloud_connection()
    if not conn:
        return
    c = conn.cursor()
    c.execute("DELETE FROM report_logs WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def get_all_users():
    conn = get_sqlitecloud_connection()
    if not conn:
        return []
    c = conn.cursor()
    c.execute("SELECT id, username, role, full_name FROM users")
    users = c.fetchall()
    conn.close()
    return users

def verify_user(username, password):
    conn = get_sqlitecloud_connection()
    if not conn:
        return None, None, False
    c = conn.cursor()
    c.execute(
        "SELECT id, password_hash, role, force_password_change FROM users WHERE username=?",
        (username,)
    )
    row = c.fetchone()
    conn.close()
    if row and pbkdf2_sha256.verify(password, row[1]):
        return row[0], row[2], bool(row[3])
    return None, None, False

def change_password(user_id, new_password):
    conn = get_sqlitecloud_connection()
    if not conn:
        return
    c = conn.cursor()
    pw = pbkdf2_sha256.hash(new_password)
    c.execute(
        "UPDATE users SET password_hash=?, force_password_change=0 WHERE id=?",
        (pw, user_id)
    )
    conn.commit()
    conn.close()

def log_report_generation(user_id, duration):
    conn = get_sqlitecloud_connection()
    if not conn:
        return
    c = conn.cursor()
    c.execute(
        "INSERT INTO report_logs(user_id,duration) VALUES (?,?)",
        (user_id, duration)
    )
    conn.commit()
    conn.close()

def get_report_summary():
    conn = get_sqlitecloud_connection()
    if not conn:
        return []
    c = conn.cursor()
    c.execute("""
        SELECT u.username,
               AVG(r.duration) AS avg_dur,
               SUM(r.duration) AS total_dur,
               COUNT(r.id)    AS cnt
          FROM report_logs r
          JOIN users u ON r.user_id = u.id
      GROUP BY u.username
    """)
    data = c.fetchall()
    conn.close()
    return data

# -------------- App‑DB Setup -------------
def create_db_engine():
    """إنشاء محرك اتصال بقاعدة البيانات."""
    try:
        server = "52.48.117.197"
        database = "R1029"
        username = "sa"
        password = "Argus@NEG"
        driver = "ODBC Driver 17 for SQL Server"
        connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=Yes;Connection Timeout=30"
        encoded_connection = quote_plus(connection_string)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_connection}")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, None
    except Exception as e:
        return None, str(e)


# ----------------- Data Fetching -----------------
@st.cache_data(ttl=600)
def fetch_data(query, params=None):
    engine, error = create_db_engine()
    if error:
        st.error(f"❌ Database connection failed: {error}")
        return None
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except SQLAlchemyError as e:
        st.error(f"❌ Error fetching data: {e}")
        return None
def calculate_vat(row):
    if row['currencyid'] == 2:
        return row['amount'] * 11.18
    elif row['currencyid'] == 3:
        return row['amount'] * 7.45
    return 0.0


def convert_gold(row):
    if row['reference'].startswith('S'):
        qty = row.get('qty', np.nan)
        if pd.isna(qty):
            qty = row['amount']
        if row['currencyid'] == 3:
            return qty
        elif row['currencyid'] == 2:
            return qty * 6 / 7
        elif row['currencyid'] == 14:
            return qty * 14 / 21
        elif row['currencyid'] == 4:
            return qty * 24 / 21
    else:
        if row['currencyid'] == 2:
            return row['amount'] * 6 / 7
        elif row['currencyid'] == 4:
            return row['amount'] * 24 / 21
    return row['amount']


def process_fifo(debits, credits):
    """معالجة FIFO بسيطة لإرجاع المعاملات المجمعة النهائية."""
    debits_q = deque(debits)
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
    history.extend([d for d in debits_q if d['remaining'] > 0])
    return history


def process_report(df, currency_type):
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.floor('D')
    df['paid_date'] = pd.to_datetime(df['paid_date'], errors='coerce').dt.floor('D')
    df['aging_days'] = np.where(df['paid_date'].isna(), '-',
                                (df['paid_date'] - df['date']).dt.days.fillna(0).astype(int))
    for col in ['amount', 'remaining', 'vat_amount']:
        df[col] = df[col].round(2)
    df['paid_date'] = df.apply(lambda r: r['paid_date'].strftime('%Y-%m-%d') if pd.notna(r['paid_date']) else 'Unpaid',
                               axis=1)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    suffix = '_gold' if currency_type != 1 else '_cash'
    return df.rename(columns={'date': 'date', 'reference': 'reference'}).add_suffix(suffix).rename(
        columns={f'date{suffix}': 'date', f'reference{suffix}': 'reference'})


def process_transactions(raw, discounts, extras, start_date):
    if raw.empty:
        return pd.DataFrame()

    def calc_row(r):
        base = r['baseAmount'] + r['basevatamount']
        if pd.to_datetime(r['date']) >= start_date:
            disc = discounts.get(r['categoryid'], 0)
            extra = extras.get(r['categoryid'], 0)
            return base - (disc * r['qty']) - (extra * r['qty'])
        return base

    def group_fn(g):
        fr = g.iloc[0]
        ref, cur, orig = fr['reference'], fr['currencyid'], fr['amount']
        if ref.startswith('S') and cur == 1:
            valid = g[~g['baseAmount'].isna()].copy()
            valid['final'] = valid.apply(calc_row, axis=1)
            amt = valid['final'].sum()
        else:
            amt = orig
        return pd.Series({'date': fr['date'], 'reference': ref,
                          'currencyid': cur, 'amount': amt, 'original_amount': orig})

    grp = raw.groupby(['functionid', 'recordid', 'date', 'reference', 'currencyid', 'amount'])
    txs = grp.apply(group_fn).reset_index(drop=True)
    txs['date'] = pd.to_datetime(txs['date'])
    txs['converted'] = txs.apply(convert_gold, axis=1)
    return txs


def calculate_aging_reports(transactions):
    """حساب تقرير Aging المُجمّع باستخدام FIFO."""
    cash_debits, cash_credits, gold_debits, gold_credits = [], [], [], []
    transactions['vat_amount'] = transactions.apply(calculate_vat, axis=1)
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    for _, r in transactions.iterrows():
        entry = {'date': r['date'], 'reference': r['reference'],
                 'amount': abs(r['converted']), 'remaining': abs(r['converted']),
                 'paid_date': None, 'vat_amount': r['vat_amount']}
        if r['currencyid'] == 1:
            (cash_debits if r['amount'] > 0 else cash_credits).append(entry)
        else:
            (gold_debits if r['amount'] > 0 else gold_credits).append(entry)
    cash = process_fifo(sorted(cash_debits, key=lambda x: x['date']), cash_credits)
    gold = process_fifo(sorted(gold_debits, key=lambda x: x['date']), gold_credits)
    cash_df = process_report(pd.DataFrame(cash), 1)
    gold_df = process_report(pd.DataFrame(gold), 2)
    df = pd.merge(cash_df, gold_df, on=['date', 'reference'], how='outer').fillna({
        'amount_gold': 0, 'remaining_gold': 0, 'paid_date_gold': '-', 'aging_days_gold': '-', 'vat_amount_gold': 0,
        'amount_cash': 0, 'remaining_cash': 0, 'paid_date_cash': '-', 'aging_days_cash': '-', 'vat_amount_cash': 0,

    })
    return df[['date', 'reference', 'amount_gold', 'remaining_gold', 'paid_date_gold', 'aging_days_gold',
               'amount_cash', 'remaining_cash', 'paid_date_cash', 'aging_days_cash'
               ]]


# ----------------- New Function: Detailed FIFO Processing -----------------
def process_fifo_detailed(debits, credits):
    """
    Simulate FIFO and record each payment application event as a separate event.
    Each event includes:
      - date: invoice date
      - reference: invoice reference
      - currencyid: 1 for cash, otherwise gold
      - invoice_amount: original invoice amount
      - applied: applied amount for this event
      - remaining: remaining balance after this event
      - paid_date: the date when this payment was applied (None if not paid)
      - aging_days: days between invoice and payment (or between invoice and today if unpaid)
    """
    # Only consider debits with date >= 01/01/2023
    debits = [d for d in debits if d['date'] >= pd.to_datetime("2023-01-01")]
    debits_q = deque(debits)
    detailed = []
    sorted_credits = sorted(credits, key=lambda x: x['date'])
    for credit in sorted_credits:
        if credit['date'] < pd.to_datetime("2023-01-01"):
            continue
        rem_credit = credit['amount']
        while rem_credit > 0 and debits_q:
            d = debits_q[0]
            payment = min(rem_credit, d['remaining'])
            d['remaining'] -= payment
            rem_credit -= payment
            event = {
                'date': d['date'],
                'reference': d['reference'],
                'currencyid': d['currencyid'],
                'invoice_amount': d['amount'],
                'Payment': payment,
                'remaining': d['remaining'],
                'paid_date': credit['date'],
                'aging_days': (credit['date'] - d['date']).days
            }
            detailed.append(event)
            if d['remaining'] == 0:
                debits_q.popleft()
    today = pd.Timestamp(datetime.now().date())
    while debits_q:
        d = debits_q.popleft()
        event = {
            'date': d['date'],
            'reference': d['reference'],
            'currencyid': d['currencyid'],
            'invoice_amount': d['amount'],
            'payment': 0,
            'remaining': d['remaining'],
            'paid_date': None,
            'aging_days': (today - d['date']).days
        }
        detailed.append(event)
    return detailed
#----------------------------------------------
def show_override_selector(raw, start_dt, key="overrides"):
    if raw is None or raw.empty:
        #st.info("لا توجد عمليات متاحة للاختيار (plantid=56 & functionid=3103) بعد تاريخ البدء.")
        return []

    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    mask = (
            (raw['plantid'] == 56) &
            (raw['functionid'] == 3103) &
            (raw['date'] > start_dt)
    )

    subset = raw.loc[mask]


    # إنشاء تسميات تحتوي على بيانات رقمية فقط
    labels = subset.apply(
        lambda r: f"{r['functionid']}|{r['recordid']}|{r['date'].date()}|{r['amount']}|{r['reference']}",
        axis=1
    ).tolist()

    return st.multiselect(
        ""
        ,
        labels,
        format_func=lambda x: f"Reference: {x.split('|')[4]} - Date: {x.split('|')[2]} - Amount: {x.split('|')[3]}",
        key=key
    )


def apply_overrides(raw, start_dt, chosen):
    for label in chosen:
        try:
            # تفكيك البيانات بشكل آمن
            parts = label.split('|')
            if len(parts) != 5:
                continue

            fid = int(parts[0])  # functionid
            rid = int(parts[1])  # recordid

            # تحديث التاريخ
            raw.loc[
                (raw['functionid'] == fid) &
                (raw['recordid'] == rid),
                'date'
            ] = start_dt
        except Exception as e:
            st.error(f"خطأ في معالجة العملية: {label} - {str(e)}")
            continue

    return raw
# ----------------- PDF Export Function -----------------
try:
    from fpdf import FPDF
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError as e:
    st.error(f"Missing required package: {e}")


def reshape_text(text):
    """Properly reshape and format Arabic text"""
    if not isinstance(text, str):
        text = str(text)
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        print(f"Text reshaping error: {e}")
        return text


def export_pdf(report_df, params):
    """توليد PDF مع دعم اللغة العربية وعرض أكبر"""
    pdf = FPDF(orientation='L')
    pdf.add_page()

    # — load fonts —
    pdf.add_font('DejaVu',   '',  'DejaVuSans.ttf',        uni=True)
    pdf.add_font('DejaVu', 'B',  'DejaVuSans-Bold.ttf',   uni=True)

    # — title in bold —
    pdf.set_font('DejaVu', 'B', 14)
pdf.cell(0, 15, reshape_text("تقرير الخصومات"), ln=1, align='C')

# استخراج اسم العميل إذا موجود وعرضه بشكل منفصل
customer_name = params.pop("اسم العميل", None)

# إعداد قائمة المعلمات مع إدراج صف فارغ في الموضع 5
params_list = list(params.items())
params_list.insert(5, ("", ""))

# تقسيم إلى النصفين الأيسر والأيمن
left_params = params_list[:5]
right_params = params_list[5:]

# طباعة اسم العميل في سطر كامل إذا كان موجودًا
if customer_name:
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, reshape_text(customer_name), ln=1, align='C')
    pdf.ln(5)

# إعداد أبعاد الأعمدة
col_width = pdf.w / 2 - 15  # عرض كل عمود مع هامش

# طباعة المعلمات في عمودين متوازيين
max_rows = max(len(left_params), len(right_params))
for i in range(max_rows):
    # بداية السطر
    pdf.set_x(10)  # بداية العمود الأيسر
    
    # العمود الأيسر
    if i < len(left_params):
        key, value = left_params[i]
        if key:  # تجاهل الصفوف الفارغة
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(col_width, 10, reshape_text(key), align='L')
            pdf.set_font('DejaVu', '', 12)
            pdf.cell(0, 10, str(value), ln=0, align='L')
    
    # الانتقال للعمود الأيمن
    pdf.set_x(pdf.w / 2 + 5)  # بداية العمود الأيمن
    
    # العمود الأيمن
    if i < len(right_params):
        key, value = right_params[i]
        if key:  # تجاهل الصفوف الفارغة
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(col_width, 10, reshape_text(key), align='L')
            pdf.set_font('DejaVu', '', 12)
            pdf.cell(0, 10, str(value), ln=0, align='L')
    
    pdf.ln()  # الانتقال للسطر التالي

pdf.ln(10)

    # إعداد أبعاد الأعمدة الجديدة (تم زيادة العرض بنسبة 30%)
    col_widths = [
        30,  # التاريخ
        40,  # الرقم المرجعي
        30,  # المبلغ النقدي
        35,  # المتبقي نقدي
        30,  # تاريخ الدفع
        35,  # أيام التقادم
        32,  # المبلغ ذهب
        30,  # المتبقي ذهب
        25  # أيام التقادم
    ]

    # إضافة ترويسة الجدول مع خلفية ملونة
    pdf.set_fill_color(200, 220, 255)
    for width, header in zip(col_widths, [
        "التاريخ",
        "الرقم المرجعي",
        "ذهب عيار 21",
        "تاريخ سداد الذهب",
        "المبلغ النقدي",
        "تاريخ سداد النقدية",
        "أيام سداد الذهب",
        "أيام سداد النقدية",

    ]):
        pdf.cell(width, 10, reshape_text(header), border=1, fill=True, align='C')
    pdf.ln()

    # إضافة بيانات الجدول
    for _, row in report_df.iterrows():
        # معالجة البيانات لتنسيق الأرقام
        amount_cash = f"{float(str(row['amount_cash']).replace(',', '')):,.2f}" if row['amount_cash'] else '0.00'
        remaining_cash = f"{float(str(row['remaining_cash']).replace(',', '')):,.2f}" if row[
            'remaining_cash'] else '0.00'
        amount_gold = f"{float(str(row['amount_gold']).replace(',', '')):,.2f}" if row['amount_gold'] else '0.000'
        remaining_gold = f"{float(str(row['remaining_gold']).replace(',', '')):,.2f}" if row[
            'remaining_gold'] else '0.000'

        for width, col in zip(col_widths, [
            str(row['date']),
            str(row['reference']),
            amount_gold,
            str(row['paid_date_gold']),
            amount_cash,
            str(row['paid_date_cash']),
            str(row['aging_days_gold']),
            str(row['aging_days_cash'])

        ]):
            pdf.cell(width, 7, reshape_text(col), border=1, align='C')
        pdf.ln()

    # — output and normalize to bytes —
    pdf_raw = pdf.output(dest='S')
    if isinstance(pdf_raw, bytearray):
        return bytes(pdf_raw)
    # otherwise it's a str
    return pdf_raw.encode('latin-1')

# ----------------- Authentication Components -----------------
def login_form():
    st.title("🔐 Invoice Aging System")
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            uid, role, force = verify_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.user_id = uid
                st.session_state.username = username
                st.session_state.role = role
                st.session_state.force_password_change = force
                return True
            else:
                st.error("Invalid username or password")
                return False
    return False


def password_change_form():
    st.title("🔑 Change Your Password")
    st.write("You must change your password before continuing.")
    with st.form("ChangePassword"):
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Update Password"):
            if not new_pw or new_pw != confirm_pw:
                st.error("Passwords do not match or are empty.")
            else:
                change_password(st.session_state.user_id, new_pw)
                st.success("Password updated! Please log in again.")
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                rerun()


# ----------------- User Management Interface -----------------
def user_management():
    st.sidebar.header("👥 User Management")
    with st.sidebar.expander("➕ Add New User"):
        with st.form("Add User"):
            new_username = st.text_input("Username", key="new_user")
            new_password = st.text_input("Password", type="password", key="new_pass")
            new_role = st.selectbox("Role", ["admin", "user"], key="new_role")
            new_fullname = st.text_input("Full Name", key="new_name")
            if st.form_submit_button("Create User"):
                if create_user(new_username, new_password, new_role, new_fullname):
                    st.success("✅ User created successfully. They will be prompted to change password on first login.")
                else:
                    st.error("❌ Username already exists")
    with st.sidebar.expander("🔄 Reset User Password"):
        users = get_all_users()
        options = [f"{u[1]} ({u[3]})" for u in users]
        selected = st.selectbox("Select user", options, key="reset_user")
        new_pw = st.text_input("New Password", type="password", key="reset_pw")
        if st.button("Reset Password"):
            uid = [u[0] for u in users if f"{u[1]} ({u[3]})" == selected][0]
            if new_pw:
                reset_user_password(uid, new_pw)
                st.success("✅ Password reset. User must change password at next login.")
            else:
                st.error("Enter a new password to reset.")
    with st.sidebar.expander("➖ Remove User"):
        users = get_all_users()
        if users:
            user_list = [f"{u[1]} ({u[3]})" for u in users if u[1] != st.session_state.username]
            selected_user = st.selectbox("Select user to remove", user_list, key="del_user")
            if st.button("Delete User"):
                user_id = [u[0] for u in users if f"{u[1]} ({u[3]})" == selected_user][0]
                delete_user(user_id)
                rerun()
        else:
            st.write("No users to display")
    with st.sidebar.expander("📊 Report Generation Summary"):
        summary = get_report_summary()
        if summary:
            df = pd.DataFrame(
                summary,
                columns=['Username', 'Average Duration (s)', 'Total Duration (s)', 'Number of Reports']
            )
            st.dataframe(df)
        else:
            st.write("No logs available.")


# ----------------- Main Application -----------------
def main_app():
    if st.session_state.get('force_password_change', False):
        password_change_form()
        return

    if st.session_state.role == "admin":
        user_management()
    with st.sidebar:
        st.write(f"👤 Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            rerun()

    st.title("📊 Aging Report")
    aging_threshold = st.sidebar.number_input("Enter Aging Days Threshold", min_value=0, value=30, step=1)

    groups = fetch_data("SELECT recordid, name FROM figrp ORDER BY name")
    if groups is None or groups.empty:
        st.error("❌ No groups found or an error occurred while fetching groups.")
        return

    customers = fetch_data("SELECT recordid, name, reference FROM fiacc WHERE groupid = 1")
    cust_list = ["Select Customer..."] + [f"{r['name']} ({r['reference']})" for _, r in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer Name", cust_list)
    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    st.sidebar.header("Category Discounts")
    discount_50 = st.sidebar.number_input("احجار عيار 21", 0.0, 1000.0, 0.0)
    discount_61 = st.sidebar.number_input("سادة عيار 21", 0.0, 1000.0, 0.0)
    discount_47 = st.sidebar.number_input("ذهب مشغول عيار 18", 0.0, 1000.0, 0.0)
    discount_62 = st.sidebar.number_input("سادة عيار 18", 0.0, 1000.0, 0.0)
    discount_48 = st.sidebar.number_input("ستار 18", 0.0, 1000.0, 0.0)
    discount_45 = st.sidebar.number_input("تعجيل دفع عيار 21", 0.0, 1000.0, 0.0)
    discount_46 = st.sidebar.number_input("تعجيل دفع عيار 18", 0.0, 1000.0, 0.0)

    # Fetch transaction data early if a customer is selected
    raw = None
    if selected_customer != "Select Customer...":
        cid = int(customers.iloc[cust_list.index(selected_customer) - 1]['recordid'])
        query = """
            SELECT f.plantid, f.functionid, f.recordid, f.date, f.reference,
                   f.currencyid, f.amount, s.qty, s.baseAmount, s.basevatamount, ivit.categoryid
            FROM fitrx f
            LEFT JOIN satrx s ON f.functionid=s.functionid AND f.recordid=s.recordid
            LEFT JOIN ivit ON s.itemid=ivit.recordid
            WHERE f.accountid = :acc
        """
        raw = fetch_data(query, {"acc": cid})
        if raw is None or raw.empty:
            st.warning("لا توجد معاملات متاحة للعميل المحدد.")
            raw = None

    # Show override selector BEFORE Generate Report
    st.markdown("### اختر العمليات خزينة الخصومات:")
    overrides = show_override_selector(raw, pd.to_datetime(start_date), key="overrides_pre_generate")

    if st.sidebar.button("Generate Report"):
        if selected_customer == "Select Customer...":
            st.error("الرجاء اختيار عميل.")
            return
        if raw is None or raw.empty:
            st.warning("لا توجد معاملات للعميل المحدد.")
            return

        start_time = time.time()  # Start measuring time
        discounts = {50: discount_50, 47: discount_47, 61: discount_61, 62: discount_62, 48: discount_48}
        extras = {
            50: discount_45,
            61: discount_45,
            47: discount_46,
            62: discount_46
        }

        # Apply overrides using the selected transactions
        raw2 = raw.copy()
        raw2 = apply_overrides(raw2, pd.to_datetime(start_date), overrides)

        # Process transactions
        txs = process_transactions(raw2, discounts, extras, pd.to_datetime(start_date))
        if txs.empty:
            st.warning("لا توجد معاملات بعد المعالجة.")
            return

        # Generate Aggregated Aging Report
        report = calculate_aging_reports(txs)
        # Filter report for transactions with date >= 01/01/2023
        report = report[pd.to_datetime(report['date']) >= pd.to_datetime("2023-01-01")]
        report['date_dt'] = pd.to_datetime(report['date'])
        report = report[(report['date_dt'] >= pd.to_datetime(start_date)) & (report['date_dt'] <= pd.to_datetime(end_date))]
        report = report.sort_values(by=['date_dt', 'paid_date_cash', 'paid_date_gold'],
                                   ascending=[True, True, True]).reset_index(drop=True)
        report = report.drop(columns=['date_dt'])
        # Format amounts with two decimals and thousands separator
        for col in ['amount_cash', 'remaining_cash', 'amount_gold', 'remaining_gold']:
            report[col] = report[col].apply(lambda x: f"{x:,.2f}")

        end_time = time.time()
        duration = end_time - start_time
        log_report_generation(st.session_state.user_id, duration)

        def highlight_row(row):
            styles = [''] * len(row)
            try:
                cash = int(row['aging_days_cash']) if row['aging_days_cash'] != '-' else 0
                gold = int(row['aging_days_gold']) if row['aging_days_gold'] != '-' else 0
            except:
                cash = gold = 0
            if cash > aging_threshold and gold > aging_threshold:
                styles = ['background-color: #FFCCCB'] * len(row)
            else:
                if cash > aging_threshold:
                    idx = row.index.get_loc('aging_days_cash')
                    styles[idx] = 'background-color: #FFCCCB'
                if gold > aging_threshold:
                    idx = row.index.get_loc('aging_days_gold')
                    styles[idx] = 'background-color: #FFCCCB'
            return styles

        styled_report = report.style.apply(highlight_row, axis=1)
        st.subheader("Aging Report")
        st.dataframe(styled_report, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col2:
            # Prepare PDF export
            report_params = {
                "اسم العميل": reshape_text(selected_customer),
                "تاريخ البداية": str(start_date),
                "تاريخ النهاية": str(end_date),
                "فترة سداد العميل": aging_threshold,
                "احجار عيار 21": discount_50,
                "سادة عيار 21": discount_61,
                "ذهب مشغول عيار 18": discount_47,
                "سادة عيار 18": discount_62,
                "ستار 18": discount_48,
                "تعجيل دفع عيار 21": discount_45,
                "تعجيل دفع عيار 18": discount_46
            }
            pdf_bytes = export_pdf(report, report_params)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download Report",
                    data=pdf_bytes,
                    file_name="تقرير_الخصومات.pdf",
                    mime="application/pdf"
                )

        # Detailed Installments Search by Reference
        st.markdown("---")
        st.subheader("تفاصيل سداد فاتورة معينة")

        # Build detailed FIFO events for installments
        cash_debits, cash_credits, gold_debits, gold_credits = [], [], [], []
        fioba = fetch_data(
            "SELECT fiscalYear, currencyid, amount FROM fioba WHERE fiscalYear = 2023 AND accountId = :acc",
            {"acc": cid}
        )
        if fioba is not None and not fioba.empty:
            for _, r in fioba.iterrows():
                entry_date = pd.to_datetime(f"{int(r['fiscalYear'])}-01-01")
                conv = r['amount']
                if r['currencyid'] != 1:
                    conv = convert_gold({'reference': '', 'amount': r['amount'], 'currencyid': r['currencyid']})
                entry = {
                    'date': entry_date,
                    'reference': 'Opening-Balance-2023',
                    'currencyid': r['currencyid'],
                    'amount': abs(conv),
                    'remaining': abs(conv)
                }
                if entry_date >= pd.to_datetime("2023-01-01"):
                    if conv >= 0:
                        if r['currencyid'] == 1:
                            cash_debits.append(entry)
                        else:
                            gold_debits.append(entry)
                    else:
                        if r['currencyid'] == 1:
                            cash_credits.append({'date': entry_date, 'amount': abs(conv)})
                        else:
                            gold_credits.append({'date': entry_date, 'amount': abs(conv)})
        for _, r in txs.iterrows():
            if r['date'] < pd.to_datetime("2023-01-01"):
                continue
            entry = {
                'date': r['date'],
                'reference': r['reference'],
                'currencyid': r['currencyid'],
                'amount': abs(r['converted']),
                'remaining': abs(r['converted'])
            }
            if r['amount'] > 0:
                if r['currencyid'] == 1:
                    cash_debits.append(entry)
                else:
                    gold_debits.append(entry)
            else:
                if r['currencyid'] == 1:
                    cash_credits.append({'date': r['date'], 'amount': abs(r['converted'])})
                else:
                    gold_credits.append({'date': r['date'], 'amount': abs(r['converted'])})
        cash_details = process_fifo_detailed(sorted(cash_debits, key=lambda x: x['date']),
                                            sorted(cash_credits, key=lambda x: x['date']))
        gold_details = process_fifo_detailed(sorted(gold_debits, key=lambda x: x['date']),
                                            sorted(gold_credits, key=lambda x: x['date']))
        cash_details_df = pd.DataFrame(cash_details)
        gold_details_df = pd.DataFrame(gold_details)

        if not cash_details_df.empty:
            cash_details_df['date'] = pd.to_datetime(cash_details_df['date'])
            cash_details_df = cash_details_df[(cash_details_df['date'] >= pd.to_datetime(start_date)) &
                                             (cash_details_df['date'] <= pd.to_datetime(end_date))]
        if not gold_details_df.empty:
            gold_details_df['date'] = pd.to_datetime(gold_details_df['date'])
            gold_details_df = gold_details_df[(gold_details_df['date'] >= pd.to_datetime(start_date)) &
                                             (gold_details_df['date'] <= pd.to_datetime(end_date))]
        # Format amounts for display
        if not cash_details_df.empty:
            cash_details_df['Remaining %'] = cash_details_df.apply(
                lambda r: (r['remaining'] / r['invoice_amount'] * 100) if r['invoice_amount'] != 0 else 0, axis=1
            )
            cash_details_df['invoice_amount'] = cash_details_df['invoice_amount'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['Payment'] = cash_details_df['Payment'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['remaining'] = cash_details_df['remaining'].apply(lambda x: f"{x:,.2f}")
            cash_details_df['Remaining %'] = cash_details_df['Remaining %'].apply(lambda x: f"{x:,.2f}")
        if not gold_details_df.empty:
            gold_details_df['Remaining %'] = gold_details_df.apply(
                lambda r: (r['remaining'] / r['invoice_amount'] * 100) if r['invoice_amount'] != 0 else 0, axis=1
            )
            gold_details_df['invoice_amount'] = gold_details_df['invoice_amount'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['Payment'] = gold_details_df['Payment'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['remaining'] = gold_details_df['remaining'].apply(lambda x: f"{x:,.2f}")
            gold_details_df['Remaining %'] = gold_details_df['Remaining %'].apply(lambda x: f"{x:,.2f}")
        if not cash_details_df.empty:
            cash_details_df['Invoice Date'] = cash_details_df['date'].dt.strftime('%Y-%m-%d')
            cash_details_df['Paid Date'] = cash_details_df['paid_date'].apply(
                lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else "Unpaid")
        if not gold_details_df.empty:
            gold_details_df['Invoice Date'] = gold_details_df['date'].dt.strftime('%Y-%m-%d')
            gold_details_df['Paid Date'] = gold_details_df['paid_date'].apply(
                lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else "Unpaid")
        st.markdown("### تفاصيل سداد الذهب")
        if not gold_details_df.empty:
            st.dataframe(gold_details_df[
                            ['Invoice Date', 'reference', 'invoice_amount', 'Payment', 'remaining', 'Remaining %',
                             'Paid Date', 'aging_days']
                        ].reset_index(drop=True), use_container_width=True)
        else:
            st.info("لا توجد بيانات سداد ذهباً لهذه الفاتورة.")
        st.markdown("### تفاصيل سداد النقدية")
        if not cash_details_df.empty:
            st.dataframe(cash_details_df[
                            ['Invoice Date', 'reference', 'invoice_amount', 'Payment', 'remaining', 'Remaining %',
                             'Paid Date', 'aging_days']
                        ].reset_index(drop=True), use_container_width=True)
        else:
            st.info("لا توجد بيانات سداد نقداً لهذه الفاتورة.")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_app()
    else:
        if login_form():
            st.rerun()
