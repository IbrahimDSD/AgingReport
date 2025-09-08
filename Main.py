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
                          get_overdues, build_summary_pdf, build_detailed_pdf, export_charts_to_pdf,
                          create_pie_chart, create_bar_chart, format_number, reshape_text)
from Test import (
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

# ----------------- Utility Functions -----------------
def normalize_username(username):
    """Normalize username by converting to lowercase and stripping whitespace"""
    return username.strip().lower() if username else None

# ----------------- Database Functions -----------------
@st.cache_resource
def create_user_db_engine():
    """إنشاء واختبار محرك SQLAlchemy لقاعدة بيانات المستخدمين"""
    try:
        engine = create_engine(USER_DB_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.scalar() == 1:
                return engine, None
            else:
                return None, "فشل اختبار اتصال قاعدة البيانات"
    except Exception as e:
        return None, str(e)

def initialize_user_database():
    """تهيئة قاعدة بيانات المستخدمين بالجداول المطلوبة والمستخدم الإداري الافتراضي"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"فشل الاتصال بقاعدة بيانات المستخدمين: {err}")
        return False

    try:
        with engine.connect() as conn:
            # إنشاء جدول المستخدمين مع تقييد الأدوار
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    role TEXT CHECK(role IN ('admin', 'accountant', 'sales_person')),
                    full_name TEXT,
                    password_change_required BOOLEAN DEFAULT FALSE
                )
            """))
            
            # إنشاء جدول صلاحيات التقارير
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reports_access (
                    username TEXT,
                    report_name TEXT,
                    PRIMARY KEY (username, report_name),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """))
            
            # إنشاء جدول تعيين مندوبي المبيعات
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS salesperson_access (
                    username TEXT,
                    salesperson_id INTEGER,
                    PRIMARY KEY (username, salesperson_id),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """))
            
            # التحقق من وجود عمود 'password_change_required'
            columns = [row[1] for row in conn.execute(text("PRAGMA table_info(users)"))]
            if "password_change_required" not in columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN password_change_required BOOLEAN DEFAULT FALSE"))
            
            # تهيئة المستخدم الإداري
            cnt = conn.execute(text(
                "SELECT COUNT(*) FROM users WHERE username = 'admin'"
            )).scalar()
            if cnt == 0:
                admin_hash = pbkdf2_sha256.hash("admin123")
                conn.execute(text(
                    "INSERT INTO users (username, password_hash, role, password_change_required) VALUES (:username, :password_hash, :role, :change_required)"
                ), {"username": "admin", "password_hash": admin_hash, "role": "admin", "change_required": False})
                
                # صلاحيات التقارير للمدير
                for rpt in ["aging_report", "discount_report", "collect_report"]:
                    conn.execute(text(
                        "INSERT INTO reports_access (username, report_name) VALUES (:username, :report_name)"
                    ), {"username": "admin", "report_name": rpt})
            
            conn.commit()
        return True
    except Exception as e:
        st.error(f"فشل تهيئة قاعدة بيانات المستخدمين: {e}")
        return False

# ----------------- وظائف المصادقة -----------------
def check_password(username, password):
    """التحقق من صحة كلمة المرور"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
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
        st.error(f"خطأ في التحقق من بيانات الاعتماد: {e}")
        return False

def get_user_role(username):
    """الحصول على دور المستخدم"""
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
    """الحصول على صلاحيات التقارير للمستخدم"""
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
    """الحصول على جميع المستخدمين"""
    engine, err = create_user_db_engine()
    if err:
        return []

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, role FROM users ORDER BY username"))
            return [(row[0], row[1]) for row in result]
    except Exception as e:
        st.error(f"خطأ في استرجاع المستخدمين: {str(e)}")
        return []

def check_password_change_required(username):
    """التحقق إذا كان المستخدم بحاجة لتغيير كلمة المرور"""
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
    """مسح متطلب تغيير كلمة المرور"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET password_change_required = FALSE WHERE username = :username"
            ), {"username": username})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"خطأ في مسح متطلب تغيير كلمة المرور: {str(e)}")
        return False

# ----------------- وظائف الإدارة -----------------
def user_exists(username):
    """التحقق من وجود مستخدم"""
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
    """إنشاء مستخدم جديد"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
            hashed_password = pbkdf2_sha256.hash(password)
            conn.execute(text(
                "INSERT INTO users (username, password_hash, role, password_change_required) VALUES (:username, :password_hash, :role, :change_required)"
            ), {"username": username, "password_hash": hashed_password, "role": role,
                "change_required": require_password_change})

            for report in reports_access:
                conn.execute(text(
                    "INSERT INTO reports_access (username, report_name) VALUES (:username, :report_name)"
                ), {"username": username, "report_name": report})

            conn.commit()
            return True
    except Exception as e:
        st.error(f"خطأ في إنشاء المستخدم: {str(e)}")
        return False

def delete_user(username):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False

    try:
        with engine.connect() as conn:
            with conn.begin():  # Start a transaction
                # Delete from dependent tables first
                conn.execute(
                    text("DELETE FROM reports_access WHERE username = :username"),
                    {"username": username}
                )
                conn.execute(
                    text("DELETE FROM salesperson_access WHERE username = :username"),
                    {"username": username}
                )
                conn.execute(
                    text("DELETE FROM user_roles WHERE username = :username"),
                    {"username": username}
                )
                # Delete from users last
                conn.execute(
                    text("DELETE FROM users WHERE username = :username"),
                    {"username": username}
                )
            # Commit is handled by conn.begin()
        return True
    except Exception as e:
        st.error(f"Error deleting user {username}: {str(e)}")
        if "FOREIGN KEY constraint failed" in str(e):
            st.error("Cannot delete user due to related records in other tables. Please check database schema for additional foreign key constraints.")
        return False

def update_user_permissions(username, role, reports_access):
    """تحديث صلاحيات المستخدم"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
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
        st.error(f"خطأ في تحديث صلاحيات المستخدم: {str(e)}")
        return False

def update_user_role(username, new_role):
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
            with conn.begin():  # Start a transaction
                result = conn.execute(
                    text("UPDATE user_roles SET role = :role WHERE username = :username"),
                    {"username": username, "role": new_role}
                )
                if result.rowcount == 0:
                    st.warning(f"لم يتم العثور على مستخدم باسم {username} في جدول user_roles")
                    return False
            
            return True
    except Exception as e:
        st.error(f"خطأ في تحديث دور المستخدم: {str(e)}")
        return False

def get_user_salesperson_ids(username):
    """الحصول على مندوبي المبيعات المعينين للمستخدم"""
    engine, err = create_user_db_engine()
    if err:
        return []

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT salesperson_id FROM salesperson_access WHERE username = :username"
            ), {"username": username})
            return [row[0] for row in result]
    except Exception as e:
        st.error(f"خطأ في الحصول على مندوبي المبيعات المعينين: {str(e)}")
        return []

def update_user_salesperson_ids(username, salesperson_ids):
    """تحديث مندوبي المبيعات المعينين للمستخدم"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM salesperson_access WHERE username = :username"
            ), {"username": username})
            
            for sp_id in salesperson_ids:
                conn.execute(text(
                    "INSERT INTO salesperson_access (username, salesperson_id) VALUES (:username, :sp_id)"
                ), {"username": username, "sp_id": sp_id})
            
            conn.commit()
            return True
    except Exception as e:
        st.error(f"خطأ في تحديث تعيينات مندوبي المبيعات: {str(e)}")
        return False

def reset_user_password(username, new_password):
    """إعادة تعيين كلمة مرور المستخدم"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
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
        st.error(f"خطأ في إعادة تعيين كلمة المرور: {str(e)}")
        return False

def mark_password_change_required(username):
    """تعليم المستخدم بأنه بحاجة لتغيير كلمة المرور"""
    engine, err = create_user_db_engine()
    if err:
        st.error(f"خطأ في اتصال قاعدة البيانات: {err}")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET password_change_required = TRUE WHERE username = :username"
            ), {"username": username})
            conn.commit()
            return True
    except Exception as e:
        st.error(f"خطأ في تعليم تغيير كلمة المرور: {str(e)}")
        return False

# ----------------- لوحة التحكم الإدارية -----------------
def admin_panel():
    st.title("👤 Admin Panel")

    if not initialize_user_database():
        st.error("Failed to initialize user database. Please check your connection and try again.")
        return

    tab1, tab2, tab3 = st.tabs(["Users", "Create User", "Reset Password"])

    with tab1:
        st.subheader("User Management")
        users = get_all_users()

        # Search bar for usernames
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
                    st.write(f"**Report Permissions:** {', '.join(reports_access) if reports_access else 'None'}")
                with col4:
                    if username != "admin" and username != st.session_state.username:
                        if st.button("Delete", key=f"delete_{username}"):
                            if delete_user(username):
                                st.success(f"User {username} deleted successfully!")
                                # Refresh user list
                                get_all_users.clear()
                                st.rerun()

                with st.expander(f"Edit {username} Permissions"):
                    # Available roles
                    available_roles = ["admin", "accountant", "sales_person"]
                    
                    # Current role
                    current_role = get_user_role(username)
                    
                    # Role selection
                    user_role = st.selectbox(
                        "Role",
                        available_roles,
                        index=available_roles.index(current_role) if current_role in available_roles else 0,
                        key=f"edit_role_{username}"
                    )
                    
                    # Report permissions
                    available_reports = ["aging_report", "discount_report", "collect_report","summaryworksheet_report","gold_report"]
                    reports_access = get_user_reports_access(username)
                    selected_reports = st.multiselect(
                        "Report Permissions",
                        available_reports,
                        default=reports_access,
                        key=f"edit_reports_{username}"
                    )
                    
                    # Salesperson assignments (for sales_person role only)
                    if user_role == "sales_person":
                        st.subheader("Assigned Salespersons")
                        
                        # Fetch salespersons from the main database (SQL Server)
                        engine = create_db_engine()  # يجب تنفيذ هذه الدالة في مكان آخر
                        salespersons = get_salespersons(engine)
                        
                        if not salespersons.empty:
                            # Display options
                            options = [f"{row['name']} ({row['spRef']})" for _, row in salespersons.iterrows()]
                            
                            # Dictionary to map display to ID
                            display_to_id = {f"{row['name']} ({row['spRef']})": row['recordid'] for _, row in salespersons.iterrows()}
                            
                            # Current assignments
                            current_access = get_user_salesperson_ids(username)
                            
                            # Convert to display names
                            current_display = [disp for disp, sp_id in display_to_id.items() if sp_id in current_access]
                            
                            selected_display = st.multiselect(
                                "Assigned Salespersons",
                                options=options,
                                default=current_display,
                                key=f"assigned_sales_{username}"
                            )
                            
                            # Convert display names to IDs
                            selected_ids = [display_to_id[disp] for disp in selected_display if disp in display_to_id]
                            
                            # Save assignments button
                            if st.button("Save Assignments", key=f"save_assigned_sales_{username}"):
                                if update_user_salesperson_ids(username, selected_ids):
                                    st.success("Salesperson assignments updated successfully!")
                    
                    # Save changes button
                    if st.button("Save Changes", key=f"save_{username}"):
                        # Update role
                        if update_user_role(username, user_role):
                            # Update report permissions
                            if update_user_permissions(username, user_role, selected_reports):
                                st.success(f"Permissions updated for user {username}")
                                # Update session state if current user
                                if username == st.session_state.username:
                                    st.session_state.role = user_role
                                    st.session_state.reports_access = selected_reports

                st.markdown("---")

    with tab2:
        st.subheader("Create New User")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        # Role selection
        role = st.selectbox("Role", ["admin", "accountant", "sales_person"], key="new_role")
        
        # Report permissions
        available_reports = ["aging_report", "discount_report", "collect_report","summaryworksheet_report","gold_report"]
        selected_reports = st.multiselect("Report Permissions", available_reports, key="new_reports")

        # Salesperson assignments (for sales_person role only)
        selected_ids = []
        if role == "sales_person":
            st.subheader("Assigned Salespersons")
            engine = create_sasp_db_engine()
            salespersons = get_salespersons(engine)
            if not salespersons.empty:
                options = [f"{row['name']} ({row['spRef']})" for _, row in salespersons.iterrows()]
                display_to_id = {f"{row['name']} ({row['spRef']})": row['recordid'] for _, row in salespersons.iterrows()}
                selected_display = st.multiselect(
                    "Assigned Salespersons",
                    options=options,
                    key="new_salespersons"
                )
                selected_ids = [display_to_id[disp] for disp in selected_display if disp in display_to_id]

        if st.button("Create User"):
            if not new_username or not new_password:
                st.error("Username and password are required")
            elif user_exists(new_username):
                st.error("Username already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                if create_user(new_username, new_password, role, selected_reports, require_password_change=True):
                    if role == "sales_person" and selected_ids:
                        update_user_salesperson_ids(new_username, selected_ids)
                    st.success(f"User {new_username} created successfully! They will need to change their password on first login.")
                    # Refresh user list
                    get_all_users.clear()
                    st.rerun()

    with tab3:
        st.subheader("Reset User Password")
        user_list = [user[0] for user in get_all_users()]
        username_to_reset = st.selectbox("Select User", user_list, key="reset_username")
        new_password = st.text_input("New Password", type="password", key="reset_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm")

        if st.button("Reset"):
            if not new_password:
                st.error("Password is required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                if reset_user_password(username_to_reset, new_password):
                    if mark_password_change_required(username_to_reset):
                        st.success(f"Password for {username_to_reset} has been reset! They will need to change it on their next login.")
                  
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
            ), {"username": username, "password_hash": hashed_password, "role": role,
                "change_required": require_password_change})
            
            # Add to user_roles table
            conn.execute(text(
                "INSERT INTO user_roles (username, role) VALUES (:username, :role)"
            ), {"username": username, "role": role})

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
            # Delete from related tables first
            conn.execute(text(
                "DELETE FROM reports_access WHERE username = :username"
            ), {"username": username})
            conn.execute(text(
                "DELETE FROM salesperson_access WHERE username = :username"
            ), {"username": username})
            conn.execute(text(
                "DELETE FROM user_roles WHERE username = :username"
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
            # Update reports access
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

def update_user_role(username, new_role):
    if new_role not in ["admin", "accountant", "sales_person"]:
        st.error(f"Invalid role: {new_role}")
        return False

    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False
    if engine is None:
        st.error("Failed to create database engine.")
        return False

    try:
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE users SET role = :role WHERE username = :username"),
                {"username": username, "role": new_role}
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error updating user role: {str(e)}")
        return False

def get_user_salesperson_ids(username):
    """Get list of salesperson IDs assigned to a user (for sales_person role)"""
    engine, err = create_user_db_engine()
    if err:
        return []

    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT salesperson_id FROM salesperson_access "
                "WHERE username = :username AND report_name = 'assigned_salespersons'"
            ), {"username": username})
            return [row[0] for row in result]
    except Exception as e:
        st.error(f"Error getting assigned salespersons: {str(e)}")
        return []

def update_user_salesperson_ids(username, salesperson_ids):
    """Update salesperson assignments for a user (sales_person role) in the salesperson_access table."""
    import streamlit as st
    from sqlalchemy import text

    engine, err = create_user_db_engine()
    if err:
        st.error(f"Database connection error: {err}")
        return False
    if engine is None:
        st.error("Failed to create database engine.")
        return False

    try:
        with engine.connect() as conn:
            # Begin a transaction
            with conn.begin():
                # Delete existing assignments for the user and report_name
                conn.execute(
                    text("DELETE FROM salesperson_access WHERE username = :username AND report_name = :report_name"),
                    {"username": username, "report_name": "assigned_salespersons"}
                )
                # Insert new assignments
                for sp_id in salesperson_ids:
                    conn.execute(
                        text("INSERT INTO salesperson_access (username, report_name, salesperson_id) VALUES (:username, :report_name, :salesperson_id)"),
                        {"username": username, "report_name": "assigned_salespersons", "salesperson_id": sp_id}
                    )
        return True
    except Exception as e:
        st.error(f"Error updating salesperson assignments: {str(e)}")
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
    st.set_page_config(
    page_title="Login",
    page_icon="🔒",
    layout="wide"
    )
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
        # st.sidebar.warning(f"⚠️ Non-admin role: {st.session_state.role}")
    
    page = st.sidebar.radio("Go to", pages)

    if page == "Admin Panel":
        admin_panel()
    else:
        # ... rest of your existing code
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="report-select-container"><label>Select Report</label></div>',
                            unsafe_allow_html=True)
        available_reports = []
        report_mapping = {
            "aging_report": "Aging Report",
            "discount_report": "Discount Report",
            "collect_report": "Collect Report",
            "summaryworksheet_report":"Summary Worksheet Report",
            "gold_report":" تقرير ميزان الذهب "
        }
        for report in st.session_state.reports_access:
            if report in report_mapping:
                available_reports.append(report_mapping[report])

        if not available_reports:
            st.warning("You don't have access to any reports. Please contact IT.")
            return

        selected_report = st.sidebar.selectbox("", available_reports, key="report_select")

        if selected_report == "Aging Report":
            from aging_report import aging_report as run_aging
            run_aging()
        elif selected_report == "Discount Report":
            from Test import main as run_discount
            run_discount()
        elif selected_report == "Collect Report":
            from FinalCode import collections_report as run_collect
            run_collect(role=st.session_state.role, username=st.session_state.username)
        elif selected_report == "Summary Worksheet Report":
            from SummaryWorkSheet import main as run_summary
            run_summary()
        elif selected_report == " تقرير ميزان الذهب ":
            from GoldReport import main as run_goldreport
            run_goldreport()

# ---------------------------------------
def get_salespersons(engine_or_tuple):
    
    query = """
        SELECT name, recordid, spRef
        FROM sasp
        WHERE name IS NOT NULL
        ORDER BY name
    """
    if engine_or_tuple is None:
        return pd.DataFrame(columns=["name", "recordid", "spRef"])
    # unpack if tuple like (engine, err)
    if isinstance(engine_or_tuple, (list, tuple)):
        if len(engine_or_tuple) == 2:
            engine, err = engine_or_tuple
            if err:
                # engine creation failed
                st.error(f"DB engine error: {err}")
                return pd.DataFrame(columns=["name", "recordid", "spRef"])
        else:
            engine = engine_or_tuple[0]
    else:
        engine = engine_or_tuple

    # now engine should be a SQLAlchemy Engine or Connection
    try:
        # prefer opening a connection context to ensure pandas gets a DBAPI connection
        if hasattr(engine, "connect"):
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
        else:
            # engine might already be a connection-like object
            df = pd.read_sql(query, engine)
        # ensure expected columns exist
        expected = ["name", "recordid", "spRef"]
        for c in expected:
            if c not in df.columns:
                df[c] = None
        return df
    except Exception as e:
        st.error(f"Failed to read salespersons from DB: {e}")
        return pd.DataFrame(columns=["name", "recordid", "spRef"])
# -----------------------------------------------------------

def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'reports_access' not in st.session_state:
        st.session_state.reports_access = None
    if 'password_change_required' not in st.session_state:
        st.session_state.password_change_required = False
    if 'show_change_password' not in st.session_state:
        st.session_state.show_change_password = False

    # Check if user is logged in
    if not st.session_state.logged_in:
        login_interface()
    else:
        # Check if password change is required
        if st.session_state.password_change_required:
            change_password_interface()
        else:
            # Main application
            report_selection()

if __name__ == "__main__":
    main()


