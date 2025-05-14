import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import hashlib
import json
import os
import base64
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import deque
import time
import importlib
from passlib.hash import pbkdf2_sha256

from aging_report import (create_db_engine, get_salespersons, get_customers,
                          get_overdues, build_summary_pdf, build_detailed_pdf,
                          create_pie_chart, create_bar_chart, format_number, reshape_text)
from discount_report import (
    create_db_engine as discount_db_engine,
    fetch_data,
    convert_gold,
    process_fifo,
    process_report,
    process_transactions,
    calculate_aging_reports,
    reshape_text as discount_reshape_text,
    export_pdf
)

# Database Connection
USER_DB_URI = (
    "sqlitecloud://cpran7d0hz.g2.sqlite.cloud:8860/"
    "user_management.db?apikey=oUEez4Dc0TFsVVIVFu8SDRiXea9YVQLOcbzWBsUwZ78"
)

# ----------------- Database Connection for User Management -----------------
@st.cache_resource
def create_user_db_engine():
    """Create and test a SQLAlchemy engine for the user database."""
    try:
        engine = create_engine(USER_DB_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.scalar() == 1:
                return engine, None
            else:
                return None, "Database connection test failed"
    except Exception as e:
        return None, str(e)

def initialize_user_database():
    """Initialize the user database with required tables and default admin user."""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Failed to connect to user database: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    role TEXT,
                    full_name TEXT,
                    password_change_required BOOLEAN DEFAULT FALSE
                )
            """))

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reports_access (
                    username TEXT,
                    report_name TEXT,
                    PRIMARY KEY (username, report_name),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """))

            # Add password_change_required column if missing
            columns = [row[1] for row in conn.execute(text("PRAGMA table_info(users)"))]
            if "password_change_required" not in columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN password_change_required BOOLEAN DEFAULT FALSE"))

            # Initialize admin user
            cnt = conn.execute(text(
                "SELECT COUNT(*) FROM users WHERE username = 'admin'"
            )).scalar()
            if cnt == 0:
                admin_hash = pbkdf2_sha256.hash("admin123")
                conn.execute(text(
                    "INSERT INTO users (username, password_hash, role, password_change_required) VALUES (:username, :password_hash, :role, :change_required)"
                ), {"username": "admin", "password_hash": admin_hash, "role": "admin", "change_required": False})

                for rpt in ["aging_report", "discount_report", "other_report2"]:
                    conn.execute(text(
                        "INSERT INTO reports_access (username, report_name) VALUES (:username, :report_name)"
                    ), {"username": "admin", "report_name": rpt})

            conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to initialize user database: {e}")
        return False

# ----------------- Authentication Functions -----------------
def check_password(username, password):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT password_hash FROM users WHERE username = :username"
            ), {"username": username})
            row = result.fetchone()
            if row is None:
                return False
            stored_hash = row[0]
            return pbkdf2_sha256.verify(password, stored_hash)
    except Exception as e:
        st.error(f"Error validating credentials: {e}")
        return False

def get_user_role(username):
    engine, err = create_user_db_engine()
    if err:
        return "user"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT role FROM users WHERE username = :username"
            ), {"username": username})
            role = result.scalar()
            return role if role else "user"
    except Exception:
        return "user"

def get_user_reports_access(username):
    engine, err = create_user_db_engine()
    if err:
        return []

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT report_name FROM reports_access WHERE username = :username"
            ), {"username": username})
            return [row[0] for row in result]
    except Exception:
        return []

@st.cache_data
def get_all_users():
    engine, err = create_user_db_engine()
    if err:
        return []

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, role FROM users ORDER BY username"))
            return [(row[0], row[1]) for row in result]
    except Exception as e:
        st.error(f"Error retrieving users: {str(e)}")
        return []

def check_password_change_required(username):
    engine, err = create_user_db_engine()
    if err:
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT password_change_required FROM users WHERE username = :username"
            ), {"username": username})
            return result.scalar() or False
    except Exception:
        return False

def clear_password_change_requirement(username):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET password_change_required = FALSE WHERE username = :username"
            ), {"username": username})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error clearing password change requirement: {str(e)}")
        return False

# ----------------- Admin Functions -----------------
def admin_panel():
    st.title("👤 Admin Panel")

    if not initialize_user_database():
        st.error("Failed to initialize user database. Please check your connection and try again.")
        return

    tab1, tab2, tab3 = st.tabs(["Users", "Create User", "Reset Password"])

    with tab1:
        st.subheader("User Management")
        users = get_all_users()

        # Add search bar for usernames
        search_username = st.text_input("Search Username", placeholder="Enter username to search...")

        if users:
            # Filter users based on search input
            if search_username:
                users = [user for user in users if search_username.lower() in user[0].lower()]
                if not users:
                    st.warning(f"No user found with username containing '{search_username}'.")
                    return

            for username, role in users:
                col1, col2, col3, col4 = st.columns([2, 2, 3, 1])

                with col1:
                    st.write(f"**Username:** {username}")
                with col2:
                    st.write(f"**Role:** {role}")
                with col3:
                    reports_access = get_user_reports_access(username)
                    st.write(f"**Reports Access:** {', '.join(reports_access) if reports_access else 'None'}")
                with col4:
                    if username != "admin" and username != st.session_state.username:
                        if st.button("Delete", key=f"delete_{username}"):
                            delete_user(username)
                            st.success(f"User {username} deleted successfully!")
                            # Clear cache to refresh user list
                            get_all_users.clear()
                            st.rerun()

                with st.expander(f"Edit {username} Permissions"):
                    available_reports = ["aging_report", "discount_report", "other_report2"]
                    selected_reports = st.multiselect(
                        "Reports Access",
                        available_reports,
                        default=reports_access,
                        key=f"edit_reports_{username}"
                    )
                    user_role = st.selectbox(
                        "Role",
                        ["admin", "user"],
                        index=0 if role == "admin" else 1,
                        key=f"edit_role_{username}"
                    )
                    if st.button("Save Changes", key=f"save_{username}"):
                        update_user_permissions(username, user_role, selected_reports)
                        st.success(f"Permissions updated for {username}")
                        # Update session state if the current user is being edited
                        if username == st.session_state.username:
                            st.session_state.role = user_role
                            st.session_state.reports_access = selected_reports

                st.markdown("---")

    with tab2:
        st.subheader("Create New User")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        role = st.selectbox("Role", ["user", "admin"], key="new_role")
        available_reports = ["aging_report", "discount_report", "other_report2"]
        selected_reports = st.multiselect("Reports Access", available_reports, key="new_reports")

        if st.button("Create User"):
            if not new_username or not new_password:
                st.error("Username and password are required")
            elif user_exists(new_username):
                st.error("Username already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                create_user(new_username, new_password, role, selected_reports, require_password_change=True)
                st.success(f"User {new_username} created successfully! They will need to change their password on first login.")
                # Clear cache to refresh user list
                get_all_users.clear()
                st.rerun()

    with tab3:
        st.subheader("Reset User Password")
        user_list = [user[0] for user in get_all_users()]
        username_to_reset = st.selectbox("Select User", user_list, key="reset_username")
        new_password = st.text_input("New Password", type="password", key="reset_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm")

        if st.button("Reset Password"):
            if not new_password:
                st.error("Password is required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                reset_user_password(username_to_reset, new_password)
                mark_password_change_required(username_to_reset)
                st.success(f"Password for {username_to_reset} has been reset! They will need to change it on their next login.")

def user_exists(username):
    engine, err = create_user_db_engine()
    if err:
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM users WHERE username = :username"
            ), {"username": username})
            return result.scalar() > 0
    except Exception:
        return False

def create_user(username, password, role, reports_access, require_password_change=False):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            hashed_password = pbkdf2_sha256.hash(password)
            conn.execute(text(
                "INSERT INTO users (username, password_hash, role, password_change_required) VALUES (:username, :password_hash, :role, :change_required)"
            ), {"username": username, "password_hash": hashed_password, "role": role, "change_required": require_password_change})

            for report in reports_access:
                conn.execute(text(
                    "INSERT INTO reports_access (username, report_name) VALUES (:username, :report_name)"
                ), {"username": username, "report_name": report})

            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def delete_user(username):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM reports_access WHERE username = :username"
            ), {"username": username})
            conn.execute(text(
                "DELETE FROM users WHERE username = :username"
            ), {"username": username})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")
        return False

def update_user_permissions(username, role, reports_access):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET role = :role WHERE username = :username"
            ), {"username": username, "role": role})
            conn.execute(text(
                "DELETE FROM reports_access WHERE username = :username"
            ), {"username": username})
            for report in reports_access:
                conn.execute(text(
                    "INSERT INTO reports_access (username, report_name) VALUES (:username, :report_name)"
                ), {"username": username, "report_name": report})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error updating user permissions: {str(e)}")
        return False

def reset_user_password(username, new_password):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            hashed_password = pbkdf2_sha256.hash(new_password)
            conn.execute(text(
                "UPDATE users SET password_hash = :password_hash WHERE username = :username"
            ), {"username": username, "password_hash": hashed_password})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error resetting password: {str(e)}")
        return False

def mark_password_change_required(username):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET password_change_required = TRUE WHERE username = :username"
            ), {"username": username})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error marking password change required: {str(e)}")
        return False

# ----------------- Password Change Interface -----------------
def change_password_interface():
    st.title("🔑 Change Password")
    st.info("Your password has been reset or you are a new user. Please change your password to continue.")

    current_password = st.text_input("Current Password", type="password", key="current_password")
    new_password = st.text_input("New Password", type="password", key="new_password_change")
    confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password_change")

    if st.button("Change Password"):
        if not check_password(st.session_state.username, current_password):
            st.error("Current password is incorrect")
        elif new_password != confirm_password:
            st.error("New passwords do not match")
        elif len(new_password) < 6:
            st.error("New password must be at least 6 characters long")
        else:
            reset_user_password(st.session_state.username, new_password)
            clear_password_change_requirement(st.session_state.username)
            st.success("Password changed successfully! Please log in with your new password.")
            # Log the user out and redirect to login
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.reports_access = None
            st.session_state.password_change_required = False
            st.rerun()

# ----------------- Login Interface -----------------
def login_interface():
    st.title("🔒 Login")
    initialize_user_database()

    with st.form(key="login_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if check_password(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = get_user_role(username)
                st.session_state.reports_access = get_user_reports_access(username)
                st.session_state.password_change_required = check_password_change_required(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ----------------- Report Selection Interface -----------------
def report_selection():
    st.markdown("""
        <style>
        .report-select-container {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            margin-bottom: 20px;
        }
        .report-select-container label {
            font-weight: bold;
            color: #4CAF50;
            font-size: 16px;
        }
        .report-select-container select {
            background-color: #ffffff;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 5px;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🧭 Navigation")
    if st.session_state.role == "admin":
        pages = ["Reports Dashboard", "Admin Panel"]
    else:
        pages = ["Reports Dashboard"]
    page = st.sidebar.radio("Go to", pages)
    
    if page == "Admin Panel":
        admin_panel()
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="report-select-container"><label>Select Report</label></div>', unsafe_allow_html=True)
        available_reports = []
        report_mapping = {
            "aging_report": "Aging Report",
            "discount_report": "Discount Report",
            "other_report2": "Other Report 2"
        }
        for report in st.session_state.reports_access:
            if report in report_mapping:
                available_reports.append(report_mapping[report])
        
        if not available_reports:
            st.warning("You don't have access to any reports. Please contact your administrator.")
            return

        selected_report = st.sidebar.selectbox("", available_reports, key="report_select")
        
        if selected_report == "Aging Report":
            run_aging_report()
        elif selected_report == "Discount Report":
            from Test import main as run_discount
            run_discount()
        elif selected_report == "Other Report 2":
            st.info("This report is under development")

def reports_dashboard():
    st.title("📊 Reports Dashboard")
    st.markdown("Please select a report from the sidebar.")

# ----------------- Aging Report Function -----------------
def get_salespersons(engine):
    """Fetch salespersons with recordid for linking with Fiacc."""
    query = """
        SELECT name, recordid
        FROM sasp
        WHERE name IS NOT NULL
        ORDER BY name
    """
    return pd.read_sql(query, engine)

def get_customers(engine, sp_id=None, groupid=1):
    """
    Fetch customers (name, reference, keyWords) filtered by groupid and optionally by sp_id.
    """
    base_sql = (
        """
        SELECT DISTINCT
            f.name       AS name,
            f.reference  AS reference,
            f.keyWords   AS keyWords
        FROM Fiacc AS f
        LEFT JOIN sasp AS s ON f.spid = s.recordid
        WHERE f.groupid = {groupid} AND f.name IS NOT NULL
        """.format(groupid=int(groupid))
    )
    if sp_id is not None:
        base_sql += f" AND s.recordid = {int(sp_id)}"

    full_query = base_sql + " ORDER BY f.name"

    try:
        df = pd.read_sql(full_query, engine)
    except Exception as e:
        st.error(f"خطأ في استعلام العملاء: {e}")
        return pd.DataFrame()
    return df

def run_aging_report():
    st.subheader("📊 تقرير متأخرات العملاء")
    engine, err = create_db_engine()
    if err:
        st.error("خطأ في الاتصال: " + err)
        return
    with st.sidebar.expander("تحديد المعايير", expanded=True):
        sps = get_salespersons(engine)
        sp_options = ["All"] + sps["name"].tolist()
        sp_search = st.text_input("Search Sales Person (Name or Ref)", placeholder="Enter name or ref to search...", key="sp_search")
        sel = None
        show_sp_selectbox = True

        if sp_search:
            sps = sps[
                sps["name"].str.contains(sp_search, case=False, na=False) |
                sps["recordid"].astype(str).str.contains(sp_search, case=False, na=False)
            ]
            sp_options = ["All"] + sps["name"].tolist()
            if len(sps) == 1:
                sel = sps["name"].iloc[0]
                st.write(f"Selected Sales Person: {sel}")
                show_sp_selectbox = False
            elif len(sps) == 0:
                st.warning("لا يوجد مندوبين مطابقين للبحث.")
                sp_options = ["All"]

        if show_sp_selectbox:
            sel = st.selectbox("Sales Person", sp_options, key="sp_select")

        sp_id = None if sel == "All" else (int(sps.loc[sps["name"] == sel, "recordid"].iloc[0]) if not sps.empty else None)

        customers = get_customers(engine, sp_id if sel != "All" else None)
        customer_options = ["الكل"] + customers["name"].tolist()
        cust_search = st.text_input("Search Customer (Name, Ref, or KeyWords)", placeholder="Enter name, ref, or keywords to search...", key="cust_search")
        selected_customer = None
        show_cust_selectbox = True

        if cust_search:
            customers = customers[
                customers["name"].str.contains(cust_search, case=False, na=False) |
                customers["reference"].astype(str).str.contains(cust_search, case=False, na=False) |
                customers["keyWords"].astype(str).str.contains(cust_search, case=False, na=False)
            ]
            customer_options = ["الكل"] + customers["name"].tolist()
            if len(customers) == 1:
                selected_customer = customers["name"].iloc[0]
                st.write(f"Selected Customer: {selected_customer}")
                show_cust_selectbox = False
            elif len(customers) == 0:
                st.warning("لا يوجد عملاء مطابقين للبحث.")
                customer_options = ["الكل"]

        if show_cust_selectbox:
            selected_customer = st.selectbox("Customer Name", customer_options, key="cust_select")

        as_of = st.date_input("Due Date", date.today())
        grace = st.number_input("Grace Period", 0, 100, 30)
        length = st.number_input("Period Length", 1, 365, 15)
        min_gold_delay = st.number_input("Minimum Gold Delay (G21)", min_value=0.0, value=0.0, step=1.0)
        min_cash_delay = st.number_input("Minimum Cash Delay (EGP)", min_value=0.0, value=0.0, step=1.0)
        report_type = st.selectbox("Report Type", ["Summary Report", "Details Report"])
    
    col1, col2 = st.columns([3, 1])
    with col2:
        generate_button = st.button("Generate Report", use_container_width=True)
    if generate_button:
        with st.spinner("جاري توليد التقرير..."):
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

            if min_gold_delay > 0:
                summary_df = summary_df[summary_df["gold_total"] > min_gold_delay]
                detail_df = detail_df[detail_df["Customer Name"].isin(summary_df["Customer"])]
            if min_cash_delay > 0:
                summary_df = summary_df[summary_df["cash_total"] > min_cash_delay]
                detail_df = detail_df[detail_df["Customer Name"].isin(summary_df["Customer"])]

            if summary_df.empty:
                st.warning("لا توجد عملاء تتطابق متأخراتهم مع المعايير المحددة.")
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
            tab1, tab2 = st.tabs(["الرسوم البيانية", "البيانات التفصيلية"])
            with tab1:
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
            with tab2:
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
                    st.dataframe(
                        display_df.rename(columns=column_mapping),
                        use_container_width=True
                    )
                    pdf = build_summary_pdf(summary_df, sel, as_of, buckets, selected_customer, grace, length)
                    filename = f"متاخرات_{sel}_{as_of}.pdf"
                else:
                    st.subheader("تفاصيل متأخرات العملاء")
                    customers = set(summary_df["Customer"])
                    if customers:
                        st.markdown("**تفاصيل الفواتير المتأخرة (بعد فترة السماحية)**")
                        for customer in sorted(customers):
                            group = detail_df[detail_df["Customer Name"] == customer]
                            if not group.empty:
                                customer_summary = summary_df[summary_df["Customer"] == customer]
                                total_cash_due = customer_summary["total_cash_due"].iloc[0] if not customer_summary.empty else 0.0
                                total_gold_due = customer_summary["total_gold_due"].iloc[0] if not customer_summary.empty else 0.0
                                total_cash_overdue = customer_summary["cash_total"].iloc[0] if not customer_summary.empty else 0.0
                                total_gold_overdue = customer_summary["gold_total"].iloc[0] if not customer_summary.empty else 0.0
                                st.markdown(f"**العميل: {customer} (كود: {customer_summary['Code'].iloc[0] if not customer_summary.empty else '-'})**")
                                color_cash = "green" if total_cash_due <= 0 else "red"
                                color_gold = "green" if total_gold_due <= 0 else "blue"
                                st.markdown(
                                    f"<span style='color: {color_gold};'>إجمالي المديونية الذهبية: {format_number(total_gold_due)}</span> | "
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
                    filename = f"تفصيلي_{sel}_{as_of}.pdf"
                if pdf and (isinstance(pdf, (bytes, str))) and len(pdf) > 0:
                    data = pdf if isinstance(pdf, (bytes, bytearray)) else pdf.encode('latin-1')
                    st.download_button("⬇️ تحميل PDF", data, filename, "application/pdf", use_container_width=True)

# ----------------- Main App -----------------
def main():
    st.set_page_config(
        page_title="NEG Reports System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.password_change_required = False

    if not st.session_state.logged_in:
        login_interface()
    elif st.session_state.password_change_required:
        change_password_interface()
    else:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
        st.sidebar.markdown(f"**Role:** {st.session_state.role}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.reports_access = None
            st.session_state.password_change_required = False
            st.rerun()
        st.sidebar.markdown("---")
        report_selection()

if __name__ == "__main__":
    main()
