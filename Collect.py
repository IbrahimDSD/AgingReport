import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from collections import deque
from datetime import datetime, date, timedelta
import time
import logging
import io
import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
import base64
import re
from tempfile import NamedTemporaryFile
import plotly.graph_objects as go
from Main import get_user_role, get_salespersons, get_user_salesperson_ids

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
DB_CONFIG = {
    "server": "52.48.117.197",
    "database": "R1029",
    "username": "sa",
    "password": "Argus@NEG",
    "driver": "ODBC Driver 17 for SQL Server"
}


# -------------- Helper Functions --------------
def reshape_text(txt):
    """Reshape text for Arabic display."""
    if not isinstance(txt, str):
        txt = str(txt)
    try:
        reshaped_text = arabic_reshaper.reshape(txt)
        return get_display(reshaped_text)
    except Exception as e:
        logger.error(f"Error reshaping text: {e}, text={txt}")
        return str(txt)


def format_number(value):
    """Format numeric values for display."""
    try:
        value = float(value)
        if value < 0:
            return f"({abs(value):,.2f})"
        elif value == 0:
            return "-"
        else:
            return f"{value:,.2f}"
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting number: {e}, value={value}")
        return str(value)


def convert_to_21k(amount, currencyid):
    """Convert any gold amount to 21K equivalent."""
    conversions = {
        2: 18 / 21,  # 18K
        3: 1.0,  # 21K
        14: 14 / 21,  # 14K
        4: 24 / 21  # 24K
    }
    return round(amount * conversions.get(currencyid, 1.0), 2)


# -------------- Gold Conversion --------------
class CustomPDF(FPDF):
    def __init__(self, username, execution_datetime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username = username
        self.execution_datetime = execution_datetime
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=10)  # تقليل الهامش السفلي
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)

    def header(self):
        if self.page_no() == 1:
            self.set_font('DejaVu', 'B', 14)
            self.ln(3)  # تقليل المسافة

    def footer(self):
        self.set_y(-12)  # تقليل الهامش السفلي
        self.set_font('DejaVu', '', 7)  # تقليل حجم خط التذييل
        username_part = reshape_text(f"User: {self.username}")
        datetime_part = f"Generated on: {self.execution_datetime} | Page {self.page_no()}/{{nb}}"
        self.cell(0, 8, username_part, 0, 0, 'L')
        self.cell(0, 8, datetime_part, 0, 0, 'R')

def create_enhanced_pdf_report(report_title, df, column_order=None, total_amount=None, start_date=None, end_date=None,
                               is_gold=False, username=None, execution_datetime=None, hide_columns=None):
   
    if username is None:
        username = st.session_state.get('username', 'Unknown User')
    if execution_datetime is None:
        execution_time = datetime.now() + timedelta(hours=3)
        execution_datetime = execution_time.strftime('%d/%m/%Y %H:%M:%S')

    # قوائم الاستثناءات
    excluded_from_percentage_global = [
        'مرتجع المبيعات', 'صافى الخصم', 'مرتجع مبيعات', 'صافي الخصم',
        'Sales Return', 'Discount', 'SALES_RETURN', 'DISCOUNT',
        'مرتجع', 'خصم', 'سداد مقدم (مرتجع مبيعات)', 'سداد مقدم (خصم)', 
        'سداد مقدم (مرتجع مبيعات)', 'سداد مقدم (خصم)'
    ]

    excluded_from_customer_percentage = [
        'سداد مقدم (مرتجع مبيعات)', 'سداد مقدم (خصم)', 'سداد مقدم',
        'مرتجع المبيعات', 'مرتجع', 'مرتجع مبيعات', 'صافى الخصم', 'صافي الخصم'
    ]

    # تحويل df إلى DataFrame إن لم يكن كذلك
    try:
        pdf_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        pdf_df = pd.DataFrame({'Info': [str(df)]})

    # فحص وجود الأعمدة الأساسية
    if 'Customer' not in pdf_df.columns or 'Type' not in pdf_df.columns:
        st.warning("البيانات لا تحتوي على الأعمدة المطلوبة (Customer, Type)")
        return None

    # الأعمدة التي يجب إخفاؤها من العرض فقط
    # ملاحظة: عمود "سداد مقدم (مرتجع مبيعات)" سيظهر ولن يتم إخفاؤه
    columns_to_hide_in_pdf = [
        'مرتجع مبيعات',  # هذا عمود مختلف
        'سداد مقدم (خصم)'
    ]
    
    if hide_columns:
        columns_to_hide_in_pdf.extend(hide_columns)

    # دالة للتحقق من الإخفاء
    def should_hide_from_pdf(col):
        col_str = str(col).strip()
        
        # لا تخفي "سداد مقدم (مرتجع مبيعات)" نهائياً
        if col_str == 'سداد مقدم (مرتجع مبيعات)':
            return False
            
        for h in columns_to_hide_in_pdf:
            h_str = str(h).strip()
            # مطابقة دقيقة تماماً
            if h_str == col_str:
                return True
        return False

    # ترتيب الأعمدة من الفترات الأصغر إلى الأكبر
    def sort_columns(cols):
        fixed_cols = ['Customer', 'مرجع العميل', 'Type']
        fixed_existing = [c for c in fixed_cols if c in cols]
        period_cols = [c for c in cols if c not in fixed_existing]

        def get_sort_key(col):
            s = str(col)
            # أعمدة السداد المقدم تأتي أولاً (قبل < 0)
            if 'سداد مقدم' in s or 'سداد مقدّم' in s:
                if 'مرتجع' in s:
                    return -3  # سداد مقدم (مرتجع المبيعات) أو (مرتجع مبيعات) أولاً
                elif 'خصم' in s:
                    return -1.5  # سداد مقدم (خصم) سيتم إخفاؤه لكن في حالة الظهور
                else:
                    return -2  # سداد مقدم العادي ثانياً
            if '< 0' in s or ' < 0' in s or '<0' in s:
                return -1
            if '> 90' in s or '90+' in s or '>90' in s:
                return 1000
            if '-' in s and not s.startswith('-'):
                try:
                    parts = re.split(r'[-–]', s)
                    first_num = re.search(r'\d+', parts[0])
                    if first_num:
                        return int(first_num.group(0))
                except Exception:
                    pass
            if 'مرتجع' in s or 'خصم' in s:
                return 2000
            m = re.search(r'\d+', s)
            if m:
                try:
                    return int(m.group(0))
                except:
                    return 500
            return 500

        period_cols_sorted = sorted(period_cols, key=get_sort_key)
        return fixed_existing + period_cols_sorted

    all_cols = list(pdf_df.columns)
    sorted_cols = sort_columns(all_cols)
    
    # الأعمدة المرئية فقط (بدون المخفية)
    visible_cols = [c for c in sorted_cols if not should_hide_from_pdf(c)]

    # === تجميع العملاء حسب فترة السداد الأعلى ===
    def group_customers_by_top_in_each_period(df, visible_cols):
        if 'Customer' not in df.columns or 'Type' not in df.columns:
            return df, {}
        
        value_rows = df[df['Type'] == 'Value'].copy()
        percentage_rows = df[df['Type'] == 'Percentage'].copy()
        
        if value_rows.empty or percentage_rows.empty:
            return df, {}
        
        fixed_cols = ['Customer', 'مرجع العميل', 'Type']
        period_cols = [col for col in visible_cols if col not in fixed_cols]
        
        # استبعاد الأعمدة التي لا نريد عمل grouping بناءً عليها
        excluded_from_grouping = ['مرتجع المبيعات', 'صافى الخصم', 'صافي الخصم', 'سداد مقدم']
        period_cols_for_grouping = [
            col for col in period_cols 
            if not any(excl in str(col) for excl in excluded_from_grouping)
        ]
        
        # تحويل النسب إلى أرقام
        percentage_rows_clean = percentage_rows.copy()
        for col in period_cols_for_grouping:
            if col in percentage_rows_clean.columns:
                percentage_rows_clean[col] = percentage_rows_clean[col].apply(
                    lambda x: float(str(x).replace('%', '')) if x and str(x).replace('%', '').replace('.', '').replace('-', '').isdigit() else 0.0
                )
        
        # تحديد الفترة الأعلى لكل عميل (فقط من الأعمدة المسموح بها)
        customer_top_periods = {}
        
        for idx, row in percentage_rows_clean.iterrows():
            customer = row['Customer']
            max_value = -1
            top_period = None
            
            for col in period_cols_for_grouping:
                if col in row:
                    val = row[col]
                    if val > max_value:
                        max_value = val
                        top_period = col
            
            if top_period and max_value > 0:
                customer_top_periods[customer] = (top_period, max_value)
        
        # تجميع العملاء في كل فترة
        period_groups = {}
        
        for period in period_cols_for_grouping:
            period_groups[period] = []
        
        for customer, (top_period, max_val) in customer_top_periods.items():
            if top_period in period_groups:
                period_groups[top_period].append((customer, max_val))
        
        # ترتيب العملاء داخل كل فترة من الأعلى للأقل
        for period in period_groups:
            period_groups[period].sort(key=lambda x: x[1], reverse=True)
            period_groups[period] = [customer for customer, _ in period_groups[period]]
        
        return df, period_groups
    
    # تطبيق التجميع
    pdf_df_full, customer_groups_by_period = group_customers_by_top_in_each_period(pdf_df, visible_cols)

    # دالة لتحويل النصوص إلى أرقام
    def to_numeric(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        s = str(v).strip()
        if s == '':
            return None
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        s = s.replace(',', '').replace('%', '')
        m = re.search(r'-?\d+(\.\d+)?', s)
        if m:
            try:
                return float(m.group(0))
            except:
                return None
        return None

    # حفظ نسخة من الأسماء الأصلية قبل التشكيل
    customer_names_map = {}
    for idx, row in pdf_df_full.iterrows():
        original_name = row.get('Customer', '')
        if original_name and original_name not in customer_names_map:
            customer_names_map[original_name] = original_name
    
    # DataFrame للعرض (بدون تشكيل - للبحث والمطابقة)
    pdf_df_display = pdf_df_full[visible_cols].copy()
    
    # DataFrame المشكل للعرض في PDF فقط
    pdf_df_display_shaped = pdf_df_display.copy()
    for col in pdf_df_display_shaped.columns:
        if pdf_df_display_shaped[col].dtype == object:
            pdf_df_display_shaped[col] = pdf_df_display_shaped[col].apply(lambda v: reshape_text(str(v)) if pd.notna(v) else '')

    # إنشاء PDF
    pdf = CustomPDF(username=username, execution_datetime=execution_datetime, orientation='L', format='A4')
    pdf.l_margin = 4
    pdf.r_margin = 4
    pdf.t_margin = 6
    pdf.add_page()

    # العنوان
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 8, reshape_text(report_title), 0, 1, 'C')
    pdf.ln(2)

    # فترة التقرير
    if start_date and end_date:
        pdf.set_font('DejaVu', '', 10)
        pdf.cell(0, 6, reshape_text(f"فترة التقرير: من {start_date} إلى {end_date}"), 0, 1, 'C')
        pdf.ln(2)

    # إعداد الجدول
    pdf.set_font('DejaVu', 'B', 7)
    cols = visible_cols
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    customer_col_width = 35  # تم تصغير عرض عمود العميل
    ref_col_width = 15 if 'مرجع العميل' in cols else 15
    type_col_width = 10  # تم تصغير عرض عمود Type
    remaining_width = page_width - customer_col_width - ref_col_width - type_col_width

    period_cols = [c for c in cols if c not in ['Customer', 'مرجع العميل', 'Type']]
    period_col_width = remaining_width / len(period_cols) if period_cols else 20

    col_widths = []
    for col in cols:
        if col == 'Customer':
            col_widths.append(customer_col_width)
        elif col == 'مرجع العميل':
            col_widths.append(ref_col_width)
        elif col == 'Type':
            col_widths.append(type_col_width)
        else:
            col_widths.append(period_col_width)

    row_h = 6

    # حساب نسب كل عميل
    def calculate_percentages():
        customer_percentages = {}
        for idx, row in pdf_df_full.iterrows():
            if row.get('Type', '') != 'Percentage':
                customer = row.get('Customer', '')
                customer_total = 0.0
                customer_values = {}

                for col in period_cols:
                    if col in row:
                        val = row.get(col, 0)
                        numeric_val = to_numeric(val)
                        
                        if not any(excl in str(col) for excl in excluded_from_customer_percentage):
                            if numeric_val is not None:
                                customer_values[col] = abs(numeric_val)
                                customer_total += abs(numeric_val)
                            else:
                                customer_values[col] = 0.0
                        else:
                            customer_values[col] = None

                customer_percentages[customer] = {}
                for col in period_cols:
                    if any(excl in str(col) for excl in excluded_from_customer_percentage):
                        customer_percentages[customer][col] = None
                    else:
                        if customer_total > 0:
                            percentage = (customer_values.get(col, 0.0) / customer_total) * 100
                            customer_percentages[customer][col] = percentage
                        else:
                            customer_percentages[customer][col] = 0.0

        return customer_percentages

    customer_percentages = calculate_percentages()

    # رسم صفوف البيانات مجمعة حسب الفترات
    pdf.set_font('DejaVu', '', 6)
    
    # تتبع الأسماء المطبوعة لكل فترة لتجنب التكرار
    printed_customers_in_period = set()
    
    # رسم كل فترة مع عملائها
    for period in period_cols:
        customers_in_period = customer_groups_by_period.get(period, [])
        
        if not customers_in_period:
            continue
        
        # إعادة تعيين الأسماء المطبوعة لكل فترة جديدة
        printed_customers_in_period = set()
        
        # إضافة صفحة جديدة إذا لزم الأمر
        if pdf.get_y() + row_h * 5 > pdf.h - pdf.b_margin:
            pdf.add_page()
        
        # header الفترة
        pdf.ln(3)
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_fill_color(0, 0, 0)  # لون أسود
        pdf.set_text_color(255, 255, 255)  # كتابة بيضاء
        period_header = reshape_text(f"فترة السداد: {period} - عدد العملاء: {len(customers_in_period)}")
        pdf.cell(0, row_h + 2, period_header, border=1, align='C', fill=True)
        pdf.ln()
        
        # رسم header الأعمدة (Customer, Type, الفترات)
        pdf.set_font('DejaVu', 'B', 7)
        pdf.set_x(pdf.l_margin)
        for i, (w, col) in enumerate(zip(col_widths, cols)):
            pdf.set_fill_color(200, 200, 200)
            pdf.set_text_color(0, 0, 0)
            header_text = reshape_text(str(col))
            pdf.cell(w, row_h, header_text, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('DejaVu', '', 6)
        
        # رسم جميع العملاء في هذه الفترة
        for customer in customers_in_period:
            # البحث بالاسم الأصلي قبل التشكيل
            customer_rows = pdf_df_display[pdf_df_display['Customer'] == customer]
            
            if customer_rows.empty:
                continue
            
            for idx, row in customer_rows.iterrows():
                # استخدام البيانات المشكلة للعرض
                display_row = pdf_df_display_shaped.loc[idx]
                
                if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
                    pdf.add_page()
                    # إعادة رسم header الأعمدة في الصفحة الجديدة
                    pdf.set_font('DejaVu', 'B', 7)
                    pdf.set_x(pdf.l_margin)
                    for i, (w, col) in enumerate(zip(col_widths, cols)):
                        pdf.set_fill_color(200, 200, 200)
                        pdf.set_text_color(0, 0, 0)
                        header_text = reshape_text(str(col))
                        pdf.cell(w, row_h, header_text, border=1, align='C', fill=True)
                    pdf.ln()
                    pdf.set_font('DejaVu', '', 6)
    
                pdf.set_x(pdf.l_margin)
                row_type = row.get('Type', '')
    
                for i, (w, col) in enumerate(zip(col_widths, cols)):
                    # استخدام البيانات المشكلة للعرض
                    val = display_row.get(col, "")
                    s = "" if pd.isna(val) else str(val)
                    
                    # قص الأسماء الطويلة
                    if col == 'Customer':
                        max_chars = 25  # تحديد عدد الأحرف الأقصى للاسم
                        if len(s) > max_chars:
                            s = s[:max_chars] + "..."
                        # عدم تكرار الاسم
                        if customer in printed_customers_in_period:
                            s = ""
                        else:
                            printed_customers_in_period.add(customer)
                    elif col == 'Type':
                        # استبدال Value بـ V و Percentage بـ P
                        if s == 'Value':
                            s = 'V'
                        elif s == 'Percentage':
                            s = 'P'
                    else:
                        max_chars = max(3, int(w / 1))
                        if len(s) > max_chars:
                            s = s[:max_chars] + "..."
    
                    if col in ['Customer', 'مرجع العميل', 'Type']:
                        align = 'C'
                        pdf.set_fill_color(255, 255, 255)
                        pdf.set_text_color(0, 0, 0)
                        cell_fill = False
    
                    elif s.replace('%', '').replace(',', '').replace('.', '').replace('-', '').isdigit() or to_numeric(s) is not None:
                        align = 'R'

                        
                        excluded_cols = ['مرتجع المبيعات','صافي الخصم', 'صافى الخصم']
                        if row_type == 'Percentage' and any(excl in str(col) for excl in excluded_cols):
                            s = "---"
                        
                        # تلوين فقط عمود الفترة الحالية (فترة الـ grouping)
                        if col == period:
                            pdf.set_fill_color(0, 0, 0)
                            pdf.set_text_color(255, 255, 255)
                        else:
                            pdf.set_fill_color(245, 245, 245)
                            pdf.set_text_color(0, 0, 0)
                        cell_fill = True
    
                    else:
                        align = 'C'
                        pdf.set_fill_color(255, 255, 255)
                        pdf.set_text_color(0, 0, 0)
                        cell_fill = False
    
                    pdf.cell(w, row_h, s, border=1, align=align, fill=cell_fill)
    
                pdf.set_text_color(0, 0, 0)
                pdf.ln()

    # دوال Grand Total
    def add_grand_total_rows():
        if pdf.get_y() + (row_h * 4) > pdf.h - pdf.b_margin:
            pdf.add_page()

        numeric_cols = period_cols

        grand_total_values = {}
        overall_total = 0.0

        for col in numeric_cols:
            total_val = 0.0
            for idx, row in pdf_df_full.iterrows():
                if row.get('Type', '') != 'Percentage' and col in row:
                    val = row.get(col, 0)
                    numeric_val = to_numeric(val)
                    if numeric_val is not None:
                        total_val += numeric_val
                        if not any(excl in str(col) for excl in excluded_from_percentage_global):
                            overall_total += numeric_val
            grand_total_values[col] = total_val

        grand_total_percentages = {}
        for col in numeric_cols:
            if not any(excl in str(col) for excl in excluded_from_percentage_global):
                if overall_total > 0:
                    grand_total_percentages[col] = (grand_total_values.get(col, 0.0) / overall_total) * 100
                else:
                    grand_total_percentages[col] = 0.0
            else:
                grand_total_percentages[col] = None

        pdf.ln(2)
        pdf.set_x(pdf.l_margin)
        for w in col_widths:
            pdf.cell(w, 1, '', border='T', align='C')
        pdf.ln()

        # سطر Grand Total (القيم)
        pdf.set_font('DejaVu', 'B', 6)
        pdf.set_x(pdf.l_margin)
        for i, (w, col) in enumerate(zip(col_widths, cols)):
            if col == 'Customer':
                text = reshape_text("إجمالي القيم")
                pdf.set_fill_color(220, 220, 220)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            elif col == 'مرجع العميل':
                text = ""
                pdf.set_fill_color(220, 220, 220)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            elif col == 'Type':
                text = reshape_text("V")
                pdf.set_fill_color(220, 220, 220)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            else:
                value = grand_total_values.get(col, 0)
                text = "0" if value == 0 else format_number(value)
                
                # تلوين باللون الأسود مع كتابة بيضاء
                pdf.set_fill_color(0, 0, 0)
                pdf.set_text_color(255, 255, 255)

                align = 'R'
                cell_fill = True

            pdf.cell(w, row_h, text, border=1, align=align, fill=cell_fill)

        pdf.set_text_color(0, 0, 0)
        pdf.ln()

        # سطر Grand Total (النسب)
        pdf.set_x(pdf.l_margin)
        for i, (w, col) in enumerate(zip(col_widths, cols)):
            if col == 'Customer':
                text = reshape_text("إجمالي النسب")
                pdf.set_fill_color(240, 240, 240)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            elif col == 'مرجع العميل':
                text = ""
                pdf.set_fill_color(240, 240, 240)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            elif col == 'Type':
                text = reshape_text("P")
                pdf.set_fill_color(240, 240, 240)
                pdf.set_text_color(0, 0, 0)
                align = 'C'
                cell_fill = True
            else:
                if any(excl in str(col) for excl in excluded_from_percentage_global):
                    text = "---"
                    pdf.set_fill_color(245, 245, 245)
                    pdf.set_text_color(0, 0, 0)
                else:
                    percentage = grand_total_percentages.get(col, 0) or 0
                    text = f"{percentage:.1f}%" if percentage > 0 else "0%"
                    # تلوين باللون الأسود مع كتابة بيضاء
                    pdf.set_fill_color(0, 0, 0)
                    pdf.set_text_color(255, 255, 255)

                align = 'R'
                cell_fill = True

            pdf.cell(w, row_h, text, border=1, align=align, fill=cell_fill)

        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    add_grand_total_rows()

    # ملخص نهائي
    if total_amount is not None:
        pdf.ln(5)
        pdf.set_font('DejaVu', 'B', 10)
        if is_gold:
            pdf.cell(0, 8, reshape_text(f"إجمالي تحصيلات الذهب (G21): {format_number(total_amount)} جرام"), 0, 1, 'R')
        else:
            pdf.cell(0, 8, reshape_text(f"إجمالي التحصيلات النقدية: {format_number(total_amount)} جنيه"), 0, 1, 'R')

    # حفظ وإرجاع البايتس
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()
def create2_pdf_report(report_title, df, total_cash, total_gold, total_cash_returns, total_gold_returns, total_discount,
                       start_date, end_date, detailed_dfs=None):
    execution_time = datetime.now() + timedelta(hours=3)
    execution_datetime = execution_time.strftime('%d/%m/%Y %H:%M:%S')
    username = st.session_state.get('username', 'Unknown User')

    pdf = CustomPDF(username=username, execution_datetime=execution_datetime)
    pdf.add_page()

    # Title
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, reshape_text(report_title), 0, 1, 'C')
    pdf.ln(5)

    # Report period
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 8, reshape_text(f"فترة التقرير : From {start_date} To {end_date}"), 0, 1, 'C')
    pdf.ln(5)

    # Summary totals
    pdf.cell(0, 8, reshape_text(f" (G21) صافى تحصيلات الذهب : {format_number(total_gold)} "), 0, 1, 'R')
    pdf.cell(0, 8, reshape_text(f" صافى تحصيلات النقدية : {format_number(total_cash)} "), 0, 1, 'R')
    pdf.ln(8)

    # Aggregated aging table
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, reshape_text("Aging Summary"), 0, 1, 'C')
    col_widths = [45, 45, 30, 45, 30]
    headers = ['فترات العمر ', 'تحصيل الذهب G21', 'G21 %', 'تحصيل كاش ', 'Cash %']
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, reshape_text(h), 1, 0, 'C')
    pdf.ln()

    pdf.set_font('DejaVu', '', 10)
    for _, row in df.iterrows():
        for w, col in zip(col_widths, df.columns):
            text = reshape_text(str(row[col]))
            pdf.cell(w, 8, text, 1, 0, 'C')
        pdf.ln()

    # Output to bytes
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()


# -------------- Database Setup --------------
def create_db_engine():
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
        logger.info("Database connection established successfully")
        return engine, None
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None, str(e)


# -------------- Data Fetching --------------
@st.cache_data(ttl=600, hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def fetch_data(query, params=None, debug_reference=None):
    engine, error = create_db_engine()
    if error:
        st.error(f"❌ Database connection failed: {error}")
        logger.error(f"Database connection failed: {error}")
        return None
    try:
        logger.debug(f"Executing query: {query}")
        logger.debug(f"Parameters: {params}")
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
            logger.info(f"Fetched {len(df)} rows from query: {query[:100]}...")
            if debug_reference and not df.empty:
                debug_row = df[df['reference'] == debug_reference]
                if not debug_row.empty:
                    logger.debug(f"Found transaction with reference {debug_reference}: {debug_row.to_dict('records')}")
                else:
                    logger.debug(f"No transaction found with reference {debug_reference}")
            return df
    except SQLAlchemyError as e:
        st.error(f"❌ Error fetching data: {e}")
        logger.error(f"Error fetching data: {e}, query={query}, params={params}")
        return None


def fetch_transactions_in_batches(accountids, start_date, end_date, batch_size=100):
    results = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    accountids = [int(acc) for acc in accountids]
    for i in range(0, len(accountids), batch_size):
        batch = accountids[i:i + batch_size]
        placeholders = ','.join(f':p{j}' for j in range(len(batch)))
        query = f"""
            SELECT f.date, f.reference, f.currencyid, f.amount, f.accountid, f.FUNCTIONID, f.plantid
            FROM fitrx f
            WHERE f.accountid IN ({placeholders})
        """
        params = {f'p{j}': acc for j, acc in enumerate(batch)}
        params['start_date'] = start_date
        params['end_date'] = end_date
        raw = fetch_data(query, params)
        if raw is not None and not raw.empty:
            results.append(raw)
        else:
            logger.warning(f"No transactions found for account batch: {batch}")
    if results:
        df = pd.concat(results, ignore_index=True)
        logger.info(f"Fetched {len(df)} transactions for accounts: {accountids[:5]}...")
        return df
    logger.warning(f"No transactions found for accounts: {accountids[:5]}...")
    return pd.DataFrame(columns=['date', 'reference', 'currencyid', 'amount', 'accountid', 'FUNCTIONID', 'plantid'])


def convert_gold(row):
    try:
        is_sales_return = row.get('FUNCTIONID') in {5103}
        converted_amount = convert_to_21k(row['amount'], row['currencyid'])
        return -abs(converted_amount) if is_sales_return else converted_amount
    except (TypeError, ValueError) as e:
        logger.error(f"Error in convert_gold: {e}, row={row}")
        return row['amount']


def is_scrap_return(row):
    """Identify scrap returns (gold/cash) based on FUNCTIONID and reference"""
    if row.get('FUNCTIONID') == 5103:  # Sales return
        reference = str(row.get('reference', '')).lower()
        if 'كسر' in reference or 'ksr' in reference:
            return True
    return False


SALES_RETURN_CATEGORY = 'مرتجع المبيعات'
DISCOUNT_CATEGORY = 'صافى الخصم'
WAITED_SALES_RETURN = 'سداد مقدم (مرتجع مبيعات)'
WAITED_DISCOUNT = 'سداد مقدم (خصم)'
WAITED_OTHER = 'سداد مقدم'
BUCKET_BINS = [-np.inf, 0, 16, 31, 46, 61, 76, 91, float('inf')]
BUCKET_LABELS = [' < 0', '0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '> 90']


def prepare_transactions(raw):
    if raw.empty:
        logger.warning("No transactions to process in prepare_transactions")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw.dropna(subset=['date'])
    raw = raw[raw['amount'] != 0]
    logger.info(f"Processing {len(raw)} transactions after cleaning")
    raw['converted'] = raw.apply(convert_gold, axis=1)
    debits = raw[raw['amount'] > 0].copy()
    credits = raw[raw['amount'] < 0].copy()
    credits['converted'] = -credits['converted']
    cash_debits = debits[debits['currencyid'] == 1].copy()
    cash_credits = credits[credits['currencyid'] == 1].copy()
    gold_debits = debits[debits['currencyid'] != 1].copy()
    gold_credits = credits[credits['currencyid'] != 1].copy()
    return cash_debits, cash_credits, gold_debits, gold_credits


def process_fifo(debits_df, credits_df, start_date=None, end_date=None, debug_reference=None, debug_accountid=None):
    if debits_df.empty or credits_df.empty:
        logger.warning(f"No debits ({len(debits_df)}) or credits ({len(credits_df)}) to process in fifo")
        return [], credits_df.to_dict('records') if not credits_df.empty else []

    debits_df['date'] = pd.to_datetime(debits_df['date'])
    credits_df['date'] = pd.to_datetime(credits_df['date'])
    start_date = pd.to_datetime(start_date) if start_date else debits_df['date'].min()
    end_date = pd.to_datetime(end_date) if end_date else debits_df['date'].max()

    has_functionid = 'FUNCTIONID' in debits_df.columns
    if has_functionid:
        prioritized_debits = debits_df[
            (debits_df['date'] >= start_date) &
            (debits_df['date'] <= end_date) &
            (debits_df['FUNCTIONID'] != 5102)
            ].copy()
        regular_debits = debits_df.drop(prioritized_debits.index).copy()
    else:
        logger.warning("Column 'FUNCTIONID' not found in debits, treating all as regular")
        prioritized_debits = pd.DataFrame()
        regular_debits = debits_df.copy()

    for df in (prioritized_debits, regular_debits):
        if not df.empty:
            df['remaining'] = df['converted']

    prioritized_q = deque(sorted(prioritized_debits.to_dict('records'), key=lambda x: x['date']))
    regular_q = deque(sorted(regular_debits.to_dict('records'), key=lambda x: x['date']))

    applications = []
    unmatched_credits = []

    for credit in sorted(credits_df.to_dict('records'), key=lambda x: x['date']):
        rem = credit['converted']
        logger.debug(f"Processing credit {credit['reference']} ({credit['date']}) amount {rem}")
        matched = False

        if start_date <= credit['date'] <= end_date and prioritized_q:
            if credit.get('FUNCTIONID', 0) != 5103 and credit.get('plantid', 0) != 56:
                while rem > 0 and prioritized_q:
                    d = prioritized_q[0]
                    apply_amt = min(rem, d['remaining'])
                    d['remaining'] -= apply_amt
                    rem -= apply_amt
                    applications.append({
                        'debit_date': d['date'],
                        'debit_reference': d['reference'],
                        'credit_date': credit['date'],
                        'credit_reference': credit['reference'],
                        'apply_amt': apply_amt,
                        'aging_days': (credit['date'] - d['date']).days,
                        'accountid': d['accountid'],
                        'currencyid': d['currencyid'],
                        'original_debit_amount': d['converted'],
                        'remaining_debit_amount': d['remaining'],
                        'credit_functionid': credit.get('FUNCTIONID', 0),
                        'credit_plantid': credit.get('plantid', 0),
                        'debit_functionid': d.get('FUNCTIONID', 0)
                    })
                    matched = True
                    if d['remaining'] <= 1e-6:
                        prioritized_q.popleft()

        while rem > 0 and regular_q:
            d = regular_q[0]
            apply_amt = min(rem, d['remaining'])
            d['remaining'] -= apply_amt
            rem -= apply_amt
            applications.append({
                'debit_date': d['date'],
                'debit_reference': d['reference'],
                'credit_date': credit['date'],
                'credit_reference': credit['reference'],
                'apply_amt': apply_amt,
                'aging_days': (credit['date'] - d['date']).days,
                'accountid': d['accountid'],
                'currencyid': d['currencyid'],
                'original_debit_amount': d['converted'],
                'remaining_debit_amount': d['remaining'],
                'credit_functionid': credit.get('FUNCTIONID', 0),
                'credit_plantid': credit.get('plantid', 0),
                'debit_functionid': d.get('FUNCTIONID', 0)
            })
            matched = True
            if d['remaining'] <= 1e-6:
                regular_q.popleft()

        if rem > 0 or not matched:
            unmatched_credit = credit.copy()
            unmatched_credit['remaining'] = rem if rem > 0 else credit['converted']
            unmatched_credit['aging_days'] = (datetime.now().date() - credit['date'].date()).days
            unmatched_credit['unmatched'] = True
            unmatched_credits.append(unmatched_credit)
            logger.warning(f"Unmatched credit {credit['reference']} remaining {unmatched_credit['remaining']}")

    for d in list(prioritized_q) + list(regular_q):
        logger.warning(f"Unmatched debit {d['reference']} remaining {d['remaining']}")

    logger.info(f"FIFO processed: {len(applications)} applications, {len(unmatched_credits)} unmatched credits")
    return applications, unmatched_credits


def create_aging_report(df, sales_return_df, discount_df):
    if df.empty and sales_return_df.empty and discount_df.empty:
        empty_df = pd.DataFrame({'Aging Period': BUCKET_LABELS, 'Collected Amount': [0.0] * len(BUCKET_LABELS)})
        return empty_df, 0.0, 0.0, 0.0

    if not df.empty:
        df['Aging Period'] = pd.cut(
            df['aging_days'],
            bins=BUCKET_BINS,
            labels=BUCKET_LABELS,
            include_lowest=False,
            right=True
        )
        summary = df.groupby('Aging Period')['apply_amt'].sum().reset_index().rename(
            columns={'apply_amt': 'Collected Amount'})
        all_buckets = pd.DataFrame({'Aging Period': BUCKET_LABELS})
        full_summary = all_buckets.merge(summary, on='Aging Period', how='left').fillna(0.0)
        total_applied = df['apply_amt'].sum()
    else:
        full_summary = pd.DataFrame({'Aging Period': BUCKET_LABELS, 'Collected Amount': [0.0] * len(BUCKET_LABELS)})
        total_applied = 0.0

    total_sales_return = sales_return_df['apply_amt'].sum() if not sales_return_df.empty else 0.0
    total_discount = discount_df['apply_amt'].sum() if not discount_df.empty else 0.0

    return full_summary, total_applied, total_sales_return, total_discount


def get_discounts(raw, start_date, end_date, accountid=None):
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    discount_negative_condition = (
            (raw['amount'] < 0) &
            (raw['plantid'] == 56) &
            (raw['date'] >= start_ts) &
            (raw['date'] <= end_ts))
    if accountid is not None:
        discount_negative_condition &= (raw['accountid'] == accountid)

    discount_negative_df = raw.loc[discount_negative_condition].copy()
    total_discount_negative = 0.0
    if not discount_negative_df.empty:
        discount_negative_df['discount_amount'] = -discount_negative_df['amount']
        total_discount_negative = discount_negative_df['discount_amount'].sum()

    discount_positive_condition = (
            (raw['amount'] > 0) &
            (raw['plantid'] == 56) &
            (raw['date'] >= start_ts) &
            (raw['date'] <= end_ts))
    if accountid is not None:
        discount_positive_condition &= (raw['accountid'] == accountid)

    discount_positive_df = raw.loc[discount_positive_condition].copy()
    total_discount_positive = 0.0
    if not discount_positive_df.empty:
        total_discount_positive = discount_positive_df['amount'].sum()

    net_discount = total_discount_negative - total_discount_positive

    return total_discount_negative, net_discount, discount_negative_df, discount_positive_df


def get_collections_report_data(cash_apps, gold_apps, raw, start_date, end_date, unmatched_credits_list,
                                accountid=None):
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    raw = raw[~raw.apply(is_scrap_return, axis=1)]

    cash_credits_raw = raw[(raw['amount'] < 0) & (raw['currencyid'] == 1) &
                           (raw['date'] >= start_ts) & (raw['date'] <= end_ts)]
    if accountid is not None:
        cash_credits_raw = cash_credits_raw[cash_credits_raw['accountid'] == accountid]
    total_cash_collections = -cash_credits_raw['amount'].sum()

    gold_credits_raw = raw[(raw['amount'] < 0) & (raw['currencyid'] != 1) &
                           (raw['date'] >= start_ts) & (raw['date'] <= end_ts)]
    if accountid is not None:
        gold_credits_raw = gold_credits_raw[gold_credits_raw['accountid'] == accountid]
    gold_credits_raw['converted'] = gold_credits_raw.apply(
        lambda r: convert_to_21k(-r['amount'], r['currencyid']), axis=1
    )
    total_gold_collections = gold_credits_raw['converted'].sum()

    total_discount_negative, net_discount, discount_negative_df, discount_positive_df = get_discounts(
        raw, start_ts, end_ts, accountid
    )

    cash_apps_all = [app for app in cash_apps
                     if start_ts <= app['credit_date'] <= end_ts
                     and (accountid is None or app['accountid'] == accountid)]
    gold_apps_all = [app for app in gold_apps
                     if start_ts <= app['credit_date'] <= end_ts
                     and (accountid is None or app['accountid'] == accountid)]

    cash_apps_normal = [
        app for app in cash_apps_all
        if app.get('credit_functionid', 0) != 5103
           and app.get('credit_plantid', 0) != 56
           and app.get('debit_functionid', 0) == 5102
    ]
    gold_apps_normal = [
        app for app in gold_apps_all
        if app.get('credit_functionid', 0) != 5103
           and app.get('credit_plantid', 0) != 56
           and app.get('debit_functionid', 0) == 5102
    ]
    cash_apps_sales_return = [app for app in cash_apps_all if app.get('credit_functionid', 0) == 5103]
    gold_apps_sales_return = [app for app in gold_apps_all if app.get('credit_functionid', 0) == 5103]
    cash_apps_discount = [app for app in cash_apps_all if app.get('credit_plantid', 0) == 56]
    gold_apps_discount = [app for app in gold_apps_all if app.get('credit_plantid', 0) == 56]

    unmatched_cash_sales_return = [app for app in unmatched_credits_list
                                   if app['currencyid'] == 1 and app.get('FUNCTIONID', 0) == 5103
                                   and start_ts <= app['date'] <= end_ts
                                   and (accountid is None or app['accountid'] == accountid)]
    unmatched_gold_sales_return = [app for app in unmatched_credits_list
                                   if app['currencyid'] != 1 and app.get('FUNCTIONID', 0) == 5103
                                   and start_ts <= app['date'] <= end_ts
                                   and (accountid is None or app['accountid'] == accountid)]
    unmatched_cash_discount = [app for app in unmatched_credits_list
                               if app['currencyid'] == 1 and app.get('plantid', 0) == 56
                               and start_ts <= app['date'] <= end_ts
                               and (accountid is None or app['accountid'] == accountid)]
    unmatched_gold_discount = [app for app in unmatched_credits_list
                               if app['currencyid'] != 1 and app.get('plantid', 0) == 56
                               and start_ts <= app['date'] <= end_ts
                               and (accountid is None or app['accountid'] == accountid)]
    unmatched_cash_other = [app for app in unmatched_credits_list
                            if app['currencyid'] == 1 and app.get('FUNCTIONID', 0) != 5103
                            and app.get('plantid', 0) != 56
                            and start_ts <= app['date'] <= end_ts
                            and (accountid is None or app['accountid'] == accountid)]
    unmatched_gold_other = [app for app in unmatched_credits_list
                            if app['currencyid'] != 1 and app.get('FUNCTIONID', 0) != 5103
                            and app.get('plantid', 0) != 56
                            and start_ts <= app['date'] <= end_ts
                            and (accountid is None or app['accountid'] == accountid)]

    total_cash_unmatched_sales_return = sum(app.get('remaining', 0) for app in unmatched_cash_sales_return)
    total_gold_unmatched_sales_return = sum(app.get('remaining', 0) for app in unmatched_gold_sales_return)
    total_cash_unmatched_discount = sum(app.get('remaining', 0) for app in unmatched_cash_discount)
    total_gold_unmatched_discount = sum(app.get('remaining', 0) for app in unmatched_gold_discount)
    total_cash_unmatched_other = sum(app.get('remaining', 0) for app in unmatched_cash_other)
    total_gold_unmatched_other = sum(app.get('remaining', 0) for app in unmatched_gold_other)

    cash_df_normal = pd.DataFrame(cash_apps_normal)
    cash_df_sales_return = pd.DataFrame(cash_apps_sales_return)
    cash_df_discount = pd.DataFrame(cash_apps_discount)
    gold_df_normal = pd.DataFrame(gold_apps_normal)
    gold_df_sales_return = pd.DataFrame(gold_apps_sales_return)
    gold_df_discount = pd.DataFrame(gold_apps_discount)

    cash_summary, total_cash_normal, cash_sales_return_amt, cash_discount_amt = create_aging_report(
        cash_df_normal, cash_df_sales_return, cash_df_discount
    )
    gold_summary, total_gold_normal, gold_sales_return_amt, gold_discount_amt = create_aging_report(
        gold_df_normal, gold_df_sales_return, gold_df_discount
    )

    cash_sales_return_amt += total_cash_unmatched_sales_return
    gold_sales_return_amt += total_gold_unmatched_sales_return
    cash_discount_amt += total_cash_unmatched_discount
    gold_discount_amt += total_gold_unmatched_discount

    special_debits = raw[
        ~raw['FUNCTIONID'].isin([5102])
        & (raw['amount'] > 0)
        & (raw['date'] >= start_ts)
        & (raw['date'] <= end_ts)
        ].copy()
    special_debits['converted'] = special_debits.apply(
        lambda r: convert_to_21k(r['amount'], r['currencyid']) if r['currencyid'] != 1 else r['amount'], axis=1
    )
    if accountid is not None:
        special_debits = special_debits[special_debits['accountid'] == accountid]

    cash_special = special_debits[special_debits['currencyid'] == 1]['amount'].sum()
    gold_special = special_debits[special_debits['currencyid'] != 1]['converted'].sum()

    total_cash = total_cash_collections - cash_sales_return_amt - cash_special - total_discount_negative
    total_gold = total_gold_collections - gold_sales_return_amt - gold_special

    combined_df = cash_summary.merge(
        gold_summary,
        on='Aging Period',
        how='left',
        suffixes=(' Cash', ' Gold (G21)')
    ).fillna(0.0)

    combined_df = pd.concat([
        combined_df,
        pd.DataFrame({
            'Aging Period': [SALES_RETURN_CATEGORY],
            'Collected Amount Cash': [cash_sales_return_amt],
            'Collected Amount Gold (G21)': [gold_sales_return_amt]
        }),
        pd.DataFrame({
            'Aging Period': [DISCOUNT_CATEGORY],
            'Collected Amount Cash': [net_discount],
            'Collected Amount Gold (G21)': [gold_discount_amt]
        }),
        pd.DataFrame({
            'Aging Period': [WAITED_SALES_RETURN],
            'Collected Amount Cash': [total_cash_unmatched_sales_return],
            'Collected Amount Gold (G21)': [total_gold_unmatched_sales_return]
        }),
        pd.DataFrame({
            'Aging Period': [WAITED_DISCOUNT],
            'Collected Amount Cash': [total_cash_unmatched_discount],
            'Collected Amount Gold (G21)': [total_gold_unmatched_discount]
        }),
        pd.DataFrame({
            'Aging Period': [WAITED_OTHER],
            'Collected Amount Cash': [total_cash_unmatched_other],
            'Collected Amount Gold (G21)': [total_gold_unmatched_other]
        }),
    ], ignore_index=True)

    combined_df = combined_df.rename(columns={
        'Aging Period': 'فترة العمر',
        'Collected Amount Cash': 'المحصل نقداً',
        'Collected Amount Gold (G21)': 'المحصل ذهباً (G21)'
    })

    aging_labels = [' < 0', '0-15', '16-30', '31-45', '46-60', '61-75', '76-90', 'Greater-Than 90']
    waited_label = 'سداد مقدم'

    combined_df['نسبة نقداً'] = 0.0
    combined_df['نسبة ذهباً'] = 0.0

    mask = combined_df['فترة العمر'].isin(aging_labels + [waited_label])

    combined_df.loc[mask, 'نسبة نقداً'] = (
            combined_df.loc[mask, 'المحصل نقداً'] / total_cash * 100
    ).round(2)
    combined_df.loc[mask, 'نسبة ذهباً'] = (
            combined_df.loc[mask, 'المحصل ذهباً (G21)'] / total_gold * 100
    ).round(2)

    return combined_df, total_cash, total_gold, cash_sales_return_amt, gold_sales_return_amt, cash_discount_amt


def debug_transaction(reference):
    query = """
        SELECT f.date, f.reference, f.currencyid, f.amount, f.accountid, f.FUNCTIONID, f.plantid
        FROM fitrx f
        WHERE f.reference = :reference
    """
    params = {'reference': reference}
    raw = fetch_data(query, params, debug_reference=reference)
    debug_info = []
    if raw is None or raw.empty:
        debug_info.append(f"Transaction {reference} not found in database.")
        return debug_info
    debug_info.append(f"Transaction {reference} found: {raw.to_dict('records')}")
    cash_debits, cash_credits, gold_debits, gold_credits = prepare_transactions(raw)
    if reference in cash_debits['reference'].values:
        debug_info.append(
            f"Transaction {reference} classified as Cash Debit: {cash_debits[cash_debits['reference'] == reference].to_dict('records')}")
    if reference in cash_credits['reference'].values:
        debug_info.append(
            f"Transaction {reference} classified as Cash Credit: {cash_credits[cash_credits['reference'] == reference].to_dict('records')}")
    if reference in gold_debits['reference'].values:
        debug_info.append(
            f"Transaction {reference} classified as Gold Debit: {gold_debits[gold_debits['reference'] == reference].to_dict('records')}")
    if reference in gold_credits['reference'].values:
        debug_info.append(
            f"Transaction {reference} classified as Gold Credit: {gold_credits[gold_credits['reference'] == reference].to_dict('records')}")
    return debug_info


def search_customer(search_term, start_date, end_date, assigned_ids=None):
    query = """
        SELECT DISTINCT f.accountid, a.name, a.keywords
        FROM fitrx f
        JOIN fiacc a ON f.accountid = a.recordid
        WHERE a.groupid = 1
        AND (a.name LIKE :search_term OR a.reference LIKE :search_term OR a.keywords LIKE :search_term)
        AND f.date BETWEEN :start_date AND :end_date
    """
    params = {'search_term': f'%{search_term}%', 'start_date': start_date, 'end_date': end_date}

    if assigned_ids:
        if not isinstance(assigned_ids, (list, tuple)) or not assigned_ids:
            st.error("Invalid or empty salesperson IDs provided.")
            logger.warning("Invalid or empty salesperson IDs provided.")
            return pd.DataFrame(columns=["accountid", "name", "keywords"])

        placeholders = ','.join([f':spid_{i}' for i in range(len(assigned_ids))])
        query += f" AND a.spid IN ({placeholders})"
        for i, spid in enumerate(assigned_ids):
            params[f'spid_{i}'] = spid

    engine = create_db_engine()
    if isinstance(engine, tuple):
        engine, err = engine
        if err:
            st.error(f"Failed to connect to database: {err}")
            logger.warning(f"Database connection error: {err}")
            return pd.DataFrame(columns=["accountid", "name", "keywords"])
    if engine is None:
        st.error("Failed to create database engine.")
        logger.warning("Failed to create database engine.")
        return pd.DataFrame(columns=["accountid", "name", "keywords"])

    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params=params)
        if result is None or result.empty:
            logger.warning(f"No customers found matching search: {search_term}")
            return pd.DataFrame(columns=["accountid", "name", "keywords"])
        result['name'] = result['name'].apply(lambda x: reshape_text(x))
        return result[["accountid", "name", 'keywords']]
    except Exception as e:
        st.error(f"Failed to search customers: {str(e)}")
        logger.error(f"Failed to search customers: {str(e)}")
        return pd.DataFrame(columns=["accountid", "name", "keywords"])


def collections_report(role=None, username=None):
    st.set_page_config(page_title="Sales Collections Report", page_icon="📊", layout="wide")
    st.title("📊 Sales Collections Report with Aging Analysis")

    if 'generate_report' not in st.session_state:
        st.session_state.generate_report = False
    if 'show_raw_main' not in st.session_state:
        st.session_state.show_raw_main = False
    if 'show_raw_customer' not in st.session_state:
        st.session_state.show_raw_customer = False
    if 'selected_customer_id' not in st.session_state:
        st.session_state.selected_customer_id = None
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = None
    if 'selected_sp' not in st.session_state:
        st.session_state.selected_sp = None

    with st.sidebar:
        user_name = st.session_state.get('current_user', username or '')
        if user_name:
            st.session_state.current_user = user_name

        today = datetime.now().date()
        start_date = st.date_input("Start Date", today.replace(day=1))
        end_date = st.date_input("End Date", today)
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

        report_type = st.selectbox("Select Report Type",
                                   ["Customer Report", "Salesperson Report", "Salesperson With Details Report"])

        if report_type == "Customer Report":
            search_term = st.text_input("Search by Customer Name or Reference or Keywords", key="customer_search")
            st.session_state.selected_sp = None
            if search_term:
                assigned_ids = None
                if role == "sales_person":
                    assigned_ids = get_user_salesperson_ids(username)
                    if not assigned_ids:
                        st.warning("No salespersons assigned to you.")
                        st.stop()
                customers_search = search_customer(search_term, start_date, end_date, assigned_ids=assigned_ids)
                if not customers_search.empty:
                    if len(customers_search) == 1:
                        st.session_state.selected_customer_id = customers_search.iloc[0]['accountid']
                        st.session_state.customer_name = customers_search.iloc[0]['name']
                        st.write(f"Customer found: {reshape_text(st.session_state.customer_name)}")
                    else:
                        customer_options = ["Select a customer..."] + [
                            f"{reshape_text(row['name'])} (ID: {row['accountid']})" for _, row in
                            customers_search.iterrows()
                        ]
                        selected_customer = st.selectbox("Select Customer", customer_options, key="customer_select")
                        if selected_customer != "Select a customer...":
                            st.session_state.selected_customer_id = int(selected_customer.split("ID: ")[1][:-1])
                            st.session_state.customer_name = selected_customer.split(" (ID:")[0]
                        else:
                            st.session_state.selected_customer_id = None
                            st.session_state.customer_name = None
                else:
                    st.warning("No customers found matching the search.")
                    st.session_state.selected_customer_id = None
                    st.session_state.customer_name = None
        else:
            engine = create_db_engine()
            if isinstance(engine, tuple):
                engine, err = engine
                if err:
                    st.error(f"Failed to connect to database: {err}")
                    st.stop()
            if engine is None:
                st.error("Failed to create database engine.")
                st.stop()
            sales_persons = get_salespersons(engine)

            if role == "sales_person":
                assigned_ids = get_user_salesperson_ids(username)
                if not assigned_ids:
                    st.warning("No salespersons assigned to you.")
                    st.stop()
                sales_persons = sales_persons[sales_persons['recordid'].isin(assigned_ids)]

            if sales_persons.empty:
                st.warning("No salespersons available.")
                st.stop()

            sp_list = ["Select a salesperson..."] + sales_persons['name'].tolist()
            selected_sp = st.selectbox("Salesperson", sp_list)
            st.session_state.selected_customer_id = None
            st.session_state.customer_name = None
            if selected_sp != "Select a salesperson...":
                st.session_state.selected_sp = selected_sp
            else:
                st.session_state.selected_sp = None

        if st.button("Generate Report", type="primary", use_container_width=True):
            if report_type == "Customer Report":
                if st.session_state.selected_customer_id:
                    st.session_state.generate_report = True
                else:
                    st.warning("Please select a customer or enter a name to search.")
            else:
                if st.session_state.selected_sp:
                    st.session_state.generate_report = True  # Fixed typo: 'true' to 'True'
                else:
                    st.warning("Please select a salesperson.")

    if st.session_state.generate_report:
        if report_type == "Customer Report":
            if st.session_state.selected_customer_id:
                accountids = [st.session_state.selected_customer_id]
                customers = pd.DataFrame(
                    {'recordid': [st.session_state.selected_customer_id], 'name': [st.session_state.customer_name]})
            else:
                st.warning("No customer selected.")
                st.stop()
        else:
            if st.session_state.selected_sp:
                spid = int(sales_persons.loc[sales_persons['name'] == st.session_state.selected_sp, 'recordid'].iloc[0])

                # جلب العملاء مع حقل المرجع مرة واحدة فقط (أداء أفضل من تشغيل استعلام لكل عميل)
                customers = fetch_data(
                    "SELECT recordid, name, ISNULL(reference, '') AS reference FROM fiacc WHERE spid = :spid and isinactive=0 " ,
                    {"spid": spid}
                )

                if customers is not None and not customers.empty:
                    # تأكد من تنسيق الأسماء والمرجع
                    customers['name'] = customers['name'].apply(lambda x: reshape_text(x) if pd.notna(x) else '')
                    customers['reference'] = customers['reference'].fillna('').astype(str)
                    accountids = customers['recordid'].astype(int).tolist()
                else:
                    st.warning("No customers found for the selected salesperson.")
                    st.stop()

        with st.spinner("Fetching and processing data... This may take a few minutes"):
            raw = fetch_transactions_in_batches(accountids, start_date, end_date)
            cash_debits, cash_credits, gold_debits, gold_credits = prepare_transactions(raw)
            cash_applications = []
            gold_applications = []
            unmatched_credits = []
            for accountid in accountids:
                cash_debits_cust = cash_debits[cash_debits['accountid'] == accountid].copy()
                cash_credits_cust = cash_credits[cash_credits['accountid'] == accountid].copy()
                gold_debits_cust = gold_debits[gold_debits['accountid'] == accountid].copy()
                gold_credits_cust = gold_credits[gold_credits['accountid'] == accountid].copy()
                cash_apps, cash_unmatched = process_fifo(cash_debits_cust, cash_credits_cust,
                                                         start_date=start_date, end_date=end_date)
                cash_applications.extend(cash_apps)
                unmatched_credits.extend(cash_unmatched)
                gold_apps, gold_unmatched = process_fifo(gold_debits_cust, gold_credits_cust,
                                                         start_date=start_date, end_date=end_date)
                gold_applications.extend(gold_apps)
                unmatched_credits.extend(gold_unmatched)
            st.session_state.cash_applications = cash_applications
            st.session_state.gold_applications = gold_applications
            st.session_state.unmatched_credits = unmatched_credits
            st.session_state.customers = customers
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.session_state.raw = raw

        if report_type == "Customer Report":
            customer_name = st.session_state.customer_name or "Unknown"
            displayed_name = reshape_text(customer_name) if customer_name != "Unknown" else "Unknown"
            st.subheader(f"📋 Customer Collections Summary:{displayed_name}")
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            combined_df, total_cash, total_gold, total_cash_returns, total_gold_returns, total_discount = get_collections_report_data(
                st.session_state.cash_applications,
                st.session_state.gold_applications,
                st.session_state.raw,
                start_ts,
                end_ts,
                st.session_state.unmatched_credits,
                accountid=st.session_state.selected_customer_id
            )

            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric(label="Total Cash Collected", value=f"{total_cash:,.2f} EGP")
            with col1:
                st.metric(label="Total Gold Collected", value=f"{total_gold:,.2f} grams (G21)")

            st.subheader("📊 Collection Details by Aging Period")
            formatted_df = combined_df.copy()
            formatted_df['المحصل نقداً'] = formatted_df['المحصل نقداً'].apply(format_number)
            formatted_df['المحصل ذهباً (G21)'] = formatted_df['المحصل ذهباً (G21)'].apply(format_number)
            formatted_df['نسبة نقداً'] = formatted_df['نسبة نقداً'].apply(lambda x: f"{x}%" if x > 0 else "-")
            formatted_df['نسبة ذهباً'] = formatted_df['نسبة ذهباً'].apply(lambda x: f"{x}%" if x > 0 else "-")
            formatted_df = formatted_df[
                ['فترة العمر', 'المحصل ذهباً (G21)', 'نسبة ذهباً', 'المحصل نقداً', 'نسبة نقداً']]
            st.dataframe(formatted_df)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', '#aec7e8']
            color_map = dict(
                zip(BUCKET_LABELS + [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN, WAITED_DISCOUNT,
                                     WAITED_OTHER], colors))
            fig_gold = go.Figure()
            for i, period in enumerate(formatted_df['فترة العمر']):
                gold_value = formatted_df.loc[i, 'المحصل ذهباً (G21)']
                if isinstance(gold_value, str):
                    try:
                        gold_value = float(gold_value.replace(',', '').replace('(', '').replace(')', ''))
                    except:
                        gold_value = 0.0
                color = color_map.get(period, '#1f77b4')
                fig_gold.add_trace(go.Bar(
                    x=[period],
                    y=[gold_value],
                    name=f'Gold - {period}',
                    marker_color=color,
                    text=[f"{gold_value:,.2f}" if gold_value > 0 else ''],
                    texttemplate='%{text}',
                    textposition='auto',
                    textfont=dict(size=14, color='white'),
                    textangle=0,
                    insidetextanchor='middle',
                    outsidetextfont=dict(size=14, color='black'),
                    showlegend=(i == 0 or period in [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN,
                                                     WAITED_DISCOUNT, WAITED_OTHER])
                ))
            fig_gold.update_layout(
                title_text=reshape_text("Gold-Collect"),
                xaxis_title=reshape_text("Aging-Period"),
                yaxis_title=reshape_text("G21-Value"),
                yaxis=dict(showticklabels=False, ticks=''),
                xaxis={'tickangle': 45},
                height=400,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_gold, use_container_width=True)

            fig_cash = go.Figure()
            for i, period in enumerate(formatted_df['فترة العمر']):
                cash_value = formatted_df.loc[i, 'المحصل نقداً']
                if isinstance(cash_value, str):
                    try:
                        cash_value = float(cash_value.replace(',', '').replace('(', '').replace(')', ''))
                    except:
                        cash_value = 0.0
                color = color_map.get(period, '#1f77b4')
                fig_cash.add_trace(go.Bar(
                    x=[period],
                    y=[cash_value],
                    name=f'Cash - {period}',
                    marker_color=color,
                    text=[f"{cash_value:,.2f}" if cash_value > 0 else ''],
                    texttemplate='%{text}',
                    textposition='auto',
                    textfont=dict(size=15),
                    textangle=0,
                    insidetextanchor='middle',
                    outsidetextfont=dict(size=14, color='black'),
                    showlegend=(i == 0 or period in [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN,
                                                     WAITED_DISCOUNT, WAITED_OTHER])
                ))
            fig_cash.update_layout(
                title_text=reshape_text("Cash-Collect"),
                xaxis_title=reshape_text("Aging-Period"),
                yaxis_title=reshape_text("Cash-Value"),
                yaxis=dict(showticklabels=False, ticks=''),
                xaxis={'tickangle': 45},
                height=400,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_cash, use_container_width=True)

            pdf_bytes = create2_pdf_report(
                report_title=f"Customer Collections Report: {reshape_text(customer_name)}",
                df=formatted_df,
                total_cash=total_cash,
                total_gold=total_gold,
                total_cash_returns=total_cash_returns,
                total_gold_returns=total_gold_returns,
                total_discount=total_discount,
                start_date=start_date,
                end_date=end_date
            )
            st.download_button(
                label="🖨️ Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Collections_Report_{reshape_text(customer_name)}_{start_date}_{end_date}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        elif report_type == "Salesperson With Details Report":
            sp_name = st.session_state.selected_sp or "Unknown"
            displayed_name = reshape_text(sp_name) if sp_name != "Unknown" else "Unknown"
            st.subheader(reshape_text(f"📋 Salesperson Collections Summary: {displayed_name}"))

            total_cash_salesperson = 0
            total_gold_salesperson = 0

            all_cash_data = []
            all_gold_data = []

            # جمع البيانات لكل عميل
            for _, customer in st.session_state.customers.iterrows():
                accountid = customer['recordid']
                customer_name = customer['name']

                # جلب مرجع العميل
                customer_ref = None
                if 'reference' in customer.index:
                    customer_ref = customer.get('reference')
                if not customer_ref or pd.isna(customer_ref):
                    try:
                        q_ref = "SELECT reference FROM fiacc WHERE recordid = :rid"
                        tmpref = fetch_data(q_ref, params={"rid": int(accountid)})
                        if tmpref is not None and not tmpref.empty and 'reference' in tmpref.columns:
                            customer_ref = tmpref.iloc[0]['reference']
                        else:
                            customer_ref = ''
                    except Exception as e:
                        logger.warning(f"Could not fetch reference for account {accountid}: {e}")
                        customer_ref = ''

                # معالجة البيانات للعميل
                cash_apps_customer = [app for app in st.session_state.cash_applications if
                                      app['accountid'] == accountid]
                gold_apps_customer = [app for app in st.session_state.gold_applications if
                                      app['accountid'] == accountid]
                unmatched_credits_customer = [cred for cred in st.session_state.unmatched_credits if
                                              cred.get('accountid') == accountid]

                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                combined_df, total_cash, total_gold, _, _, _ = get_collections_report_data(
                    cash_apps_customer,
                    gold_apps_customer,
                    st.session_state.raw,
                    start_ts,
                    end_ts,
                    unmatched_credits_customer,
                    accountid=accountid
                )

                total_cash_salesperson += total_cash
                total_gold_salesperson += total_gold

                # إنشاء سطرين لكل عميل: القيمة والنسبة
                cash_value_row = {'Customer': customer_name, 'مرجع العميل': customer_ref, 'Type': 'Value'}
                cash_percent_row = {'Customer': customer_name, 'مرجع العميل': customer_ref, 'Type': 'Percentage'}
                gold_value_row = {'Customer': customer_name, 'مرجع العميل': customer_ref, 'Type': 'Value'}
                gold_percent_row = {'Customer': customer_name, 'مرجع العميل': customer_ref, 'Type': 'Percentage'}

                for _, row in combined_df.iterrows():
                    period = row['فترة العمر']
                    cash_value = row['المحصل نقداً']
                    gold_value = row['المحصل ذهباً (G21)']

                    # حساب النسبة المئوية
                    cash_percentage = (cash_value / total_cash * 100) if total_cash_salesperson != 0 else 0
                    gold_percentage = (gold_value / total_gold * 100) if total_gold_salesperson != 0 else 0

                    cash_value_row[period] = cash_value
                    cash_percent_row[period] = cash_percentage
                    gold_value_row[period] = gold_value
                    gold_percent_row[period] = gold_percentage

                all_cash_data.extend([cash_value_row, cash_percent_row])
                all_gold_data.extend([gold_value_row, gold_percent_row])

            # ترتيب الأعمدة تنازلياً حسب الفترة
            column_order_descending = ['> 90', '76-90', '61-75', '46-60', '31-45', '16-30', '0-15', ' < 0',
                                       SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY,
                                       WAITED_SALES_RETURN, WAITED_DISCOUNT, WAITED_OTHER]

            # بناء DataFrames
            cash_df = pd.DataFrame(all_cash_data)
            gold_df = pd.DataFrame(all_gold_data)

            # التأكد من وجود جميع الأعمدة
            for c in column_order_descending:
                if c not in cash_df.columns:
                    cash_df[c] = 0.0
                if c not in gold_df.columns:
                    gold_df[c] = 0.0

            # ترتيب الأعمدة
            cols_order = ['مرجع العميل', 'Customer', 'Type'] + column_order_descending
            cash_df = cash_df[cols_order]
            gold_df = gold_df[cols_order]

            # عرض البيانات
            st.subheader("💵 Cash Collections by Customer ")
            formatted_cash = cash_df.copy()

            for i, row in formatted_cash.iterrows():
                if row['Type'] == 'Value':
                    for c in column_order_descending:
                        formatted_cash.loc[i, c] = format_number(float(row[c]) if pd.notna(row[c]) else 0.0)
                elif row['Type'] == 'Percentage':
                    for c in column_order_descending:
                        formatted_cash.loc[i, c] = f"{float(row[c]):.2f}%" if pd.notna(row[c]) and row[
                            c] != 0 else "0.00%"

            formatted_cash['Customer'] = formatted_cash['Customer'].apply(lambda x: reshape_text(x) if x else x)
            st.dataframe(formatted_cash, use_container_width=True)

            st.subheader("🥇 Gold Collections by Customer (G21)")
            formatted_gold = gold_df.copy()

            for i, row in formatted_gold.iterrows():
                if row['Type'] == 'Value':
                    for c in column_order_descending:
                        formatted_gold.loc[i, c] = format_number(float(row[c]) if pd.notna(row[c]) else 0.0)
                elif row['Type'] == 'Percentage':
                    for c in column_order_descending:
                        formatted_gold.loc[i, c] = f"{float(row[c]):.2f}%" if pd.notna(row[c]) and row[
                            c] != 0 else "0.00%"

            formatted_gold['Customer'] = formatted_gold['Customer'].apply(lambda x: reshape_text(x) if x else x)
            st.dataframe(formatted_gold, use_container_width=True)

            # تحضير PDFs مع التظليل المحسن
            pdf_bytes_cash = create_enhanced_pdf_report(
                report_title=f"Salesperson Cash Collections Report: {sp_name}",
                df=formatted_cash,
                column_order=column_order_descending,
                total_amount=total_cash_salesperson,
                start_date=start_date,
                end_date=end_date,
                is_gold=False
            )

            pdf_bytes_gold = create_enhanced_pdf_report(
                report_title=f"Salesperson Gold Collections Report: {sp_name}",
                df=formatted_gold,
                column_order=column_order_descending,
                total_amount=total_gold_salesperson,
                start_date=start_date,
                end_date=end_date,
                is_gold=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📄 Download Cash Report as PDF",
                    data=pdf_bytes_cash,
                    file_name=f"تقرير التحصيلات النقدية _{sp_name}_{start_date}_{end_date}.pdf ",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📄 Download Gold Report as PDF",
                    data=pdf_bytes_gold,
                    file_name=f"تقرير التحصيلات الذهب _{sp_name}_{start_date}_{end_date}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )



        else:
            sp_name = st.session_state.selected_sp or "Unknown"
            displayed_name = reshape_text(sp_name) if sp_name != "Unknown" else "Unknown"
            st.subheader(reshape_text(f"📋 Salesperson Collections Summary: {displayed_name}"))
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            combined_df, total_cash, total_gold, total_cash_returns, total_gold_returns, total_discount = get_collections_report_data(
                st.session_state.cash_applications,
                st.session_state.gold_applications,
                st.session_state.raw,
                start_ts,
                end_ts,
                st.session_state.unmatched_credits
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Cash Collected", value=f"{total_cash:,.2f} EGP")
            with col2:
                st.metric(label="Total Gold Collected", value=f"{total_gold:,.2f} grams (G21)")

            st.subheader("📊 Collection Details by Aging Period")
            formatted_df = combined_df.copy()
            formatted_df['المحصل نقداً'] = formatted_df['المحصل نقداً'].apply(format_number)
            formatted_df['المحصل ذهباً (G21)'] = formatted_df['المحصل ذهباً (G21)'].apply(format_number)
            formatted_df['نسبة نقداً'] = formatted_df['نسبة نقداً'].apply(lambda x: f"{x}%" if x > 0 else "-")
            formatted_df['نسبة ذهباً'] = formatted_df['نسبة ذهباً'].apply(lambda x: f"{x}%" if x > 0 else "-")
            formatted_df = formatted_df[
                ['فترة العمر', 'المحصل ذهباً (G21)', 'نسبة ذهباً', 'المحصل نقداً', 'نسبة نقداً']]
            st.dataframe(formatted_df)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', '#aec7e8']
            color_map = dict(
                zip(BUCKET_LABELS + [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN, WAITED_DISCOUNT,
                                     WAITED_OTHER], colors))
            fig_gold = go.Figure()
            for i, period in enumerate(formatted_df['فترة العمر']):
                gold_value = formatted_df.loc[i, 'المحصل ذهباً (G21)']
                if isinstance(gold_value, str):
                    try:
                        gold_value = float(gold_value.replace(',', '').replace('(', '').replace(')', ''))
                    except:
                        gold_value = 0.0
                color = color_map.get(period, '#1f77b4')
                fig_gold.add_trace(go.Bar(
                    x=[period],
                    y=[gold_value],
                    name=f'Gold - {period}',
                    marker_color=color,
                    text=[f"{gold_value:,.2f}" if gold_value > 0 else ''],
                    texttemplate='%{text}',
                    textposition='auto',
                    textfont=dict(size=14, color='white'),
                    textangle=0,
                    insidetextanchor='middle',
                    outsidetextfont=dict(size=14, color='black'),
                    showlegend=(i == 0 or period in [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN,
                                                     WAITED_DISCOUNT, WAITED_OTHER])
                ))
            fig_gold.update_layout(
                title_text=reshape_text("Gold-Collect"),
                xaxis_title=reshape_text("Aging-Period"),
                yaxis_title=reshape_text("G21-Value"),
                yaxis=dict(showticklabels=False, ticks=''),
                xaxis={'tickangle': 45},
                height=400,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_gold, use_container_width=True)

            fig_cash = go.Figure()
            for i, period in enumerate(formatted_df['فترة العمر']):
                cash_value = formatted_df.loc[i, 'المحصل نقداً']
                if isinstance(cash_value, str):
                    try:
                        cash_value = float(cash_value.replace(',', '').replace('(', '').replace(')', ''))
                    except:
                        cash_value = 0.0
                color = color_map.get(period, '#1f77b4')
                fig_cash.add_trace(go.Bar(
                    x=[period],
                    y=[cash_value],
                    name=f'Cash - {period}',
                    marker_color=color,
                    text=[f"{cash_value:,.2f}" if cash_value > 0 else ''],
                    texttemplate='%{text}',
                    textposition='auto',
                    textfont=dict(size=15),
                    textangle=0,
                    insidetextanchor='middle',
                    outsidetextfont=dict(size=14, color='black'),
                    showlegend=(i == 0 or period in [SALES_RETURN_CATEGORY, DISCOUNT_CATEGORY, WAITED_SALES_RETURN,
                                                     WAITED_DISCOUNT, WAITED_OTHER])
                ))
            fig_cash.update_layout(
                title_text=reshape_text("Cash-Collect"),
                xaxis_title=reshape_text("Aging-Period"),
                yaxis_title=reshape_text("Cash-Value"),
                yaxis=dict(showticklabels=False, ticks=''),
                xaxis={'tickangle': 45},
                height=400,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_cash, use_container_width=True)

            pdf_bytes = create2_pdf_report(
                report_title=f"Salesperson Collections Report: {sp_name}",
                df=formatted_df,
                total_cash=total_cash,
                total_gold=total_gold,
                total_cash_returns=total_cash_returns,
                total_gold_returns=total_gold_returns,
                total_discount=total_discount,
                start_date=start_date,
                end_date=end_date
            )
            st.download_button(
                label="🖨️ Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Collections_Report_{sp_name}_{start_date}_{end_date}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


    else:
        st.info("🔍 Please specify report criteria in the sidebar and click 'Generate Report'.")









