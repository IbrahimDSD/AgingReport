import streamlit as st
import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import base64
from fpdf import FPDF
import numpy as np
import io
from datetime import datetime, date
import arabic_reshaper
from bidi.algorithm import get_display

# Database Configuration
DB_CONFIG = {
    "server": "52.48.117.197",
    "database": "R1029",
    "username": "sa",
    "password": "Argus@NEG",
    "driver": "ODBC Driver 17 for SQL Server"
}

# Page configuration
st.set_page_config(
    page_title="Work Sheet Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Create database engine
def create_db_engine():
    """Create database engine with error handling"""
    try:
        connection_string = (
            f"DRIVER={DB_CONFIG['driver']};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
            f"TrustServerCertificate=Yes;"
            f"MultipleActiveResultSets=true;"
            f"Connection Timeout=30"
        )
        encoded_connection = quote_plus(connection_string)
        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={encoded_connection}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None


# Updated SQL Query with additional filters
SQL_QUERY = """
SET NOCOUNT ON;

DECLARE @FromDate DATE = :from_date
DECLARE @TODate DATE = :to_date
DECLARE @WorkSeetRef NVARCHAR(15) = :work_ref
DECLARE @JobOrderRef NVARCHAR(15) = :job_order_ref
DECLARE @WorkCenterName NVARCHAR(100) = :work_center_name
DECLARE @ItemCategory NVARCHAR(50) = :item_category
DECLARE @Metal NVARCHAR(15) = :metal

--------------------------------------------------------

CREATE TABLE #1
(
WorkSheetId INT,
WorkSeetRef NVARCHAR(15),
WorkSheetDate DATE,
PostingStatus SMALLINT,
StartTime DATETIME,
EndTime DATETIME,
JobId INT,
JobOrderRef NVARCHAR(15),
ItemRef NVARCHAR(15),
ItemName NVARCHAR(50),
WorkCenterRef NVARCHAR(15),
ItemCategory NVARCHAR(50),
Metal NVARCHAR(15),
WorkCenterName NVARCHAR(100),
Firstqty NUMERIC(13,3),
RMQty NUMERIC(13,3),
EndQty NUMERIC(13,3),
Qty_Gold NUMERIC(13,3),
Qty_Add NUMERIC(13,3),
Qty_H_Add NUMERIC(13,3)
);

INSERT INTO #1 (WorkSheetId,WorkSeetRef,WorkSheetDate,PostingStatus,StartTime,EndTime,JobId,JobOrderRef,ItemRef,ItemName,ItemCategory,Metal,WorkCenterRef,WorkCenterName,Firstqty,RMQty,EndQty)
SELECT W.recordId,W.reference,w.date,w.status, w.startTime,w.endTime,W.jobId, J.reference,i.sku, i.name,t.name, 
Metal = CASE WHEN metalId = 1 THEN 'G18' ELSE CASE WHEN metalId = 2 THEN 'G21' END END,
c.reference,c.name,Firstqty = w.wipQty, w.RMQty, EndQty = w.eopQty
FROM mfwst W
JOIN Mfjob J ON J.recordId = W.jobId
JOIN MFWCT c ON c.recordId = W.workCenterId
JOIN IVIT i ON i.recordId = j.itemId
JOIN IVCA t ON t.recordId = i.categoryId
WHERE 
w.DATE BETWEEN @FromDate AND @TODate
AND (ISNULL(@WorkSeetRef,'') = '' OR W.reference = @WorkSeetRef)
AND (ISNULL(@JobOrderRef,'') = '' OR J.reference = @JobOrderRef)
AND (ISNULL(@WorkCenterName,'') = '' OR c.name LIKE '%' + @WorkCenterName + '%')
AND (ISNULL(@ItemCategory,'') = '' OR t.name LIKE '%' + @ItemCategory + '%')
AND (ISNULL(@Metal,'') = '' OR 
     (CASE WHEN metalId = 1 THEN 'G18' ELSE CASE WHEN metalId = 2 THEN 'G21' END END) = @Metal);

---------------------------------------------------------------------------------------------

CREATE TABLE #2
(
WorkSheetId INT,
JobId INT,
ItemRef NVARCHAR(15),
ItemName NVARCHAR(50),
RawCategoryRef NVARCHAR(15),
RawCategoryName NVARCHAR(15),
isMetal SMALLINT,
Issue NUMERIC(13,3),
[Return] NUMERIC(13,3),
Loss NUMERIC(13,3)
);

INSERT INTO #2 (WorkSheetId,JobId,ItemRef,ItemName,RawCategoryRef,RawCategoryName,isMetal,Issue,[Return],Loss)
SELECT h.worksheetId,h.jobId,I.sku, I.name, r.reference,r.name,isMetal = r.isMetal,
Issue = SUM(CASE WHEN h.type = 1 THEN d.qty ELSE 0 END),
[Return] = SUM(CASE WHEN h.type = 2 THEN d.qty ELSE 0 END),
Loss = SUM(CASE WHEN h.type = 3 THEN d.qty ELSE 0 END)
FROM mfimi d
JOIN mfima h ON h.recordId = d.imaId
JOIN mfwst w ON w.recordId = h.worksheetId
JOIN IVIT i ON i.recordId = d.itemId
JOIN IVMFR m ON m.itemId = i.recordId
LEFT JOIN MFRMC r ON r.recordId = m.rmcId
WHERE w.DATE BETWEEN @FromDate AND @TODate
AND (ISNULL(@WorkSeetRef,'') = '' OR W.reference = @WorkSeetRef)
GROUP BY h.worksheetId,h.jobId,I.sku, I.name, r.reference,r.name, r.isMetal;

-----------------------------------------------------------

CREATE TABLE #3
(
WorkSheetId INT,
JobId INT,
Qty_Gold NUMERIC(13,3),
Qty_Add NUMERIC(13,3),
Qty_H_Add NUMERIC(13,3)
);

INSERT INTO #3 (WorkSheetId,JobId,Qty_Gold,Qty_Add,Qty_H_Add)
SELECT #2.WorkSheetId, #2.JobId,
Qty_Gold = ISNULL(SUM(CASE WHEN #2.isMetal = 1 THEN (#2.Issue - #2.[Return] - #2.Loss) END),0),
Qty_Add = ISNULL(SUM(CASE WHEN #2.isMetal = 0 AND #2.ItemRef <> '2060073' THEN (#2.Issue - #2.[Return] - #2.Loss) END),0),
Qty_H_Add = ISNULL(SUM(CASE WHEN #2.isMetal = 0 AND #2.ItemRef = '2060073' THEN (#2.Issue - #2.[Return] - #2.Loss) END),0)
FROM #2
GROUP BY WorkSheetId, JobId;

---- ===========================================================================

UPDATE #1 SET #1.Qty_Gold = ISNULL(#3.Qty_Gold,0), #1.Qty_Add = ISNULL(#3.Qty_Add,0), #1.Qty_H_Add = ISNULL(#3.Qty_H_Add,0) 
FROM #3
WHERE #1.WorkSheetId = #3.WorkSheetId AND #1.JobId = #3.JobId;

---- ===========================================================================

SELECT * FROM #1
ORDER BY #1.WorkCenterRef, #1.WorkSheetId, #1.WorkSheetDate;

DROP TABLE #1;
DROP TABLE #2;
DROP TABLE #3;
"""


# Function to get unique values for filters
@st.cache_data(ttl=3600)
def get_filter_options():
    """Get unique values for filter dropdowns"""
    engine = create_db_engine()
    if not engine:
        return [], [], [], []

    try:
        with engine.connect() as conn:
            # Get Job Order References
            job_orders_query = """
            SELECT DISTINCT J.reference as JobOrderRef 
            FROM Mfjob J 
            WHERE J.reference IS NOT NULL 
            ORDER BY J.reference
            """
            job_orders = pd.read_sql(job_orders_query, conn)

            # Get Work Center Names
            work_centers_query = """
            SELECT DISTINCT c.name as WorkCenterName 
            FROM MFWCT c 
            WHERE c.name IS NOT NULL 
            ORDER BY c.name
            """
            work_centers = pd.read_sql(work_centers_query, conn)

            # Get Item Categories
            categories_query = """
            SELECT DISTINCT t.name as ItemCategory 
            FROM IVCA t 
            WHERE t.name IS NOT NULL 
            ORDER BY t.name
            """
            categories = pd.read_sql(categories_query, conn)

            # Metal options are fixed
            metals = ['G18', 'G21']

            return (
                job_orders['JobOrderRef'].tolist() if not job_orders.empty else [],
                work_centers['WorkCenterName'].tolist() if not work_centers.empty else [],
                categories['ItemCategory'].tolist() if not categories.empty else [],
                metals
            )
    except Exception as e:
        st.error(f"Error fetching filter options: {str(e)}")
        return [], [], [], []


# Cache data fetching with additional parameters
@st.cache_data(ttl=600)
def fetch_data(from_date, to_date, work_ref, job_order_ref, work_center_name, item_category, metal):
    """Fetch data from SQL with caching and additional filters"""
    engine = create_db_engine()
    if not engine:
        return None

    try:
        with engine.connect() as conn:
            # Execute with parameters
            result = conn.execute(
                text(SQL_QUERY),
                {
                    "from_date": from_date,
                    "to_date": to_date,
                    "work_ref": work_ref if work_ref else "",
                    "job_order_ref": job_order_ref if job_order_ref else "",
                    "work_center_name": work_center_name if work_center_name else "",
                    "item_category": item_category if item_category else "",
                    "metal": metal if metal else ""
                }
            )
            # Fetch all results
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None


def reshape_text(text):
    """Handle Arabic text reshaping if needed"""
    return str(text) if text else ""


class CustomPDF(FPDF):
    def __init__(self, username, execution_datetime, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username = username
        self.execution_datetime = execution_datetime
        self.headers = []
        self.col_width = 0

    def header(self):
        self.set_font('DejaVu', '', 9)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, get_display(arabic_reshaper.reshape("Job Reference")), 0, 1, 'C', True)
        self.ln(2)

        # Print table headers
        if self.headers:
            self.set_font('DejaVu', 'B', 7)
            self.set_fill_color(128, 128, 128)
            self.set_text_color(255, 255, 255)
            for header in self.headers:
                self.cell(self.col_width, 6, get_display(arabic_reshaper.reshape(header))[:10], 1, 0, 'C', True)
            self.ln()
            self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Generated by {self.username} on {self.execution_datetime}', 0, 0, 'C')


def create_pdf_with_arabic_support(df, grouped=False, username="System User", execution_datetime=None):

    if execution_datetime is None:
        execution_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    font_path = r'DejaVuSans.ttf'
    font_name = 'DejaVu'

    pdf = CustomPDF(username, execution_datetime, orientation='L', unit='mm', format='A4')
    pdf.add_font(font_name, '', font_path, uni=True)
    pdf.add_font(font_name, 'B', font_path, uni=True)
    pdf.add_font(font_name, 'I', font_path, uni=True)
    pdf.set_font(font_name, '', 8)
    pdf.add_page()

    def safe_text(txt):
        if not isinstance(txt, str):
            txt = str(txt)
        try:
            reshaped_text = arabic_reshaper.reshape(txt)
            return get_display(reshaped_text)
        except Exception:
            return txt

    # استثناء الأعمدة المحددة
    exclude_columns = ['WorkSheetId', 'StartTime', 'EndTime', 'JobId', 'WorkCenterName', 'PostingStatus']
    df = df.drop(columns=[col for col in exclude_columns if col in df.columns])

    total_gold = 0
    total_add = 0

    # فرض التجميع حسب WorkCenterName
    if 'WorkCenterName' in df.columns:
        grouped_df = df.groupby('WorkCenterName')

        for work_center, group in grouped_df:
            # 1. عنوان مركز العمل
            pdf.set_font(font_name, 'B', 9)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(0, 8, f"{safe_text('Work Center')}: {safe_text(work_center)}", 0, 1, 'L', True)
            pdf.ln(1)

            # 2. ملخص المجموعة
            gold_sum = group['Qty_Gold'].sum() if 'Qty_Gold' in group.columns else 0
            add_sum = group['Qty_Add'].sum() if 'Qty_Add' in group.columns else 0
            unique_items = group['ItemRef'].nunique() if 'ItemRef' in group.columns else 0

            pdf.set_font(font_name, '', 8)
            summary_text = f"{safe_text('Worksheets')}: {len(group)} | {safe_text('Items')}: {unique_items} | {safe_text('Gold')}: {gold_sum:.3f} | {safe_text('Add')}: {add_sum:.3f}"
            pdf.cell(0, 6, summary_text, 0, 1, 'L')
            pdf.ln(1)

            # 3. رؤوس الأعمدة لكل مجموعة
            display_df = group.drop(columns=['WorkCenterName']) if 'WorkCenterName' in group.columns else group.copy()
            headers = list(display_df.columns)
            # حماية من قسمة على صفر
            if len(headers) == 0:
                continue
            col_width = (pdf.w - 20) / len(headers)
            pdf.headers = headers
            pdf.col_width = col_width

            pdf.set_font(font_name, 'B', 6)
            pdf.set_fill_color(230, 230, 230)
            for col in headers:
                pdf.cell(col_width, 5, safe_text(col), 1, 0, 'C', True)
            pdf.ln()

            # 4. صفوف البيانات
            pdf.set_font(font_name, '', 6)
            pdf.set_fill_color(245, 245, 245)
            for i, (_, row) in enumerate(display_df.iterrows()):
                fill = i % 2 == 0
                for col in headers:
                    cell_value = safe_text(str(row[col]) if not pd.isna(row[col]) else "")
                    if len(cell_value) > 12:
                        cell_value = cell_value[:9] + "..."
                    pdf.cell(col_width, 5, cell_value, 1, 0, 'C', fill)

                    # تجميع الإجماليات العامة
                    if col == 'Qty_Gold':
                        try:
                            total_gold += float(row[col]) if not pd.isna(row[col]) else 0
                        except Exception:
                            pass
                    if col == 'Qty_Add':
                        try:
                            total_add += float(row[col]) if not pd.isna(row[col]) else 0
                        except Exception:
                            pass
                pdf.ln()

            # --- هنا نضيف صف المجموع الفرعي لكل مجموعة ---
            pdf.ln(1)
            pdf.set_font(font_name, 'B', 6)
            pdf.set_fill_color(210, 235, 210)  # لون خلفية مميز للمجموع الفرعي
            # أول خلية: تسمية المجموع الفرعي
            pdf.cell(col_width, 6, safe_text("المجموع الفرعي"), 1, 0, 'C', True)

            # بقية الأعمدة: إما قيم المجموع الفرعي أو خانات فارغة
            for col in headers[1:]:
                if col == 'Qty_Gold':
                    pdf.cell(col_width, 6, f"{gold_sum:.3f}", 1, 0, 'C', True)
                elif col == 'Qty_Add':
                    pdf.cell(col_width, 6, f"{add_sum:.3f}", 1, 0, 'C', True)
                else:
                    pdf.cell(col_width, 6, "", 1, 0, 'C', True)
            pdf.ln()

            # فاصل بين المجموعات
            pdf.ln(3)
            pdf.set_fill_color(180, 180, 180)
            pdf.cell(0, 1, "", 0, 1, 'C', True)  # خط فاصل
            pdf.ln(2)
    else:
        # في حالة عدم وجود WorkCenterName
        display_df = df.copy()
        headers = list(display_df.columns)
        if len(headers) == 0:
            return bytes(pdf.output(dest='S'))
        col_width = (pdf.w - 20) / len(headers)
        pdf.headers = headers
        pdf.col_width = col_width

        pdf.set_font(font_name, 'B', 6)
        pdf.set_fill_color(230, 230, 230)
        for col in headers:
            pdf.cell(col_width, 5, safe_text(col), 1, 0, 'C', True)
        pdf.ln()

        pdf.set_font(font_name, '', 6)
        pdf.set_fill_color(245, 245, 245)
        for i, (_, row) in enumerate(display_df.iterrows()):
            fill = i % 2 == 0
            for col in headers:
                cell_value = safe_text(str(row[col]) if not pd.isna(row[col]) else "")
                if len(cell_value) > 12:
                    cell_value = cell_value[:9] + "..."
                pdf.cell(col_width, 5, cell_value, 1, 0, 'C', fill)

                if col == 'Qty_Gold':
                    try:
                        total_gold += float(row[col]) if not pd.isna(row[col]) else 0
                    except Exception:
                        pass
                if col == 'Qty_Add':
                    try:
                        total_add += float(row[col]) if not pd.isna(row[col]) else 0
                    except Exception:
                        pass
            pdf.ln()

    # صف الإجمالي العام
    pdf.ln(2)
    pdf.set_font(font_name, 'B', 7)
    pdf.set_fill_color(220, 220, 220)

    # التأكد من وجود headers
    if 'headers' in locals():
        pdf.cell(col_width, 5, safe_text("Grand Total"), 1, 0, 'C', True)
        for col in headers[1:]:
            if col == 'Qty_Gold':
                pdf.cell(col_width, 5, f"{total_gold:.3f}", 1, 0, 'C', True)
            elif col == 'Qty_Add':
                pdf.cell(col_width, 5, f"{total_add:.3f}", 1, 0, 'C', True)
            else:
                pdf.cell(col_width, 5, "", 1, 0, 'C', True)
        pdf.ln()

    return bytes(pdf.output(dest='S'))


def create_excel_download(df, grouped=False):
    """Create Excel file for download"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if grouped:
            # Create separate sheets for each work center
            work_centers = df['WorkCenterName'].unique()

            # Create summary sheet first
            summary_data = []
            for work_center in sorted(work_centers):
                center_data = df[df['WorkCenterName'] == work_center]
                summary_data.append({
                    'Work Center': work_center,
                    'Work Center Ref': center_data['WorkCenterRef'].iloc[
                        0] if 'WorkCenterRef' in center_data.columns else '',
                    'Worksheet Count': len(center_data),
                    'Unique Items': center_data['ItemRef'].nunique(),
                    'Total Gold Qty': center_data['Qty_Gold'].sum(),
                    'Total Add Qty': center_data['Qty_Add'].sum(),
                    'Total H Add Qty': center_data['Qty_H_Add'].sum(),
                    'Total First Qty': center_data['Firstqty'].sum(),
                    'Total RM Qty': center_data['RMQty'].sum(),
                    'Total End Qty': center_data['EndQty'].sum()
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Create individual sheets for each work center
            for work_center in sorted(work_centers):
                center_data = df[df['WorkCenterName'] == work_center].copy()
                center_display = prepare_export_data(center_data)

                # Clean sheet name
                sheet_name = str(work_center)[:31]
                sheet_name = ''.join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_'))

                center_display.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Single sheet with all data
            df_export = prepare_export_data(df)
            df_export.to_excel(writer, sheet_name='Work Sheet Report', index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    return output.getvalue()


def group_by_workcenter(df):
    """Group data by WorkCenterName"""
    if df.empty:
        return df

    # Check which columns exist in the dataframe
    available_columns = df.columns.tolist()

    # Columns to aggregate - only include those that exist
    agg_columns = {}

    # Numeric columns to sum
    numeric_cols = ['Firstqty', 'RMQty', 'EndQty', 'Qty_Gold', 'Qty_Add', 'Qty_H_Add']
    for col in numeric_cols:
        if col in available_columns:
            agg_columns[col] = 'sum'

    # Count worksheets
    if 'WorkSheetId' in available_columns:
        agg_columns['WorkSheetId'] = 'count'
    elif 'WorkSeetRef' in available_columns:
        agg_columns['WorkSeetRef'] = 'count'

    # Count unique items
    if 'ItemRef' in available_columns:
        agg_columns['ItemRef'] = lambda x: x.nunique()

    # Get first WorkCenterRef
    if 'WorkCenterRef' in available_columns:
        agg_columns['WorkCenterRef'] = 'first'

    # Group and aggregate
    grouped_df = df.groupby('WorkCenterName').agg(agg_columns).reset_index()

    # Rename columns
    rename_dict = {}
    if 'WorkSheetId' in grouped_df.columns:
        rename_dict['WorkSheetId'] = 'WorksheetCount'
    elif 'WorkSeetRef' in grouped_df.columns:
        rename_dict['WorkSeetRef'] = 'WorksheetCount'

    if 'ItemRef' in grouped_df.columns:
        rename_dict['ItemRef'] = 'UniqueItems'

    if rename_dict:
        grouped_df = grouped_df.rename(columns=rename_dict)

    # Reorder columns
    preferred_order = [
        'WorkCenterName', 'WorkCenterRef', 'WorksheetCount', 'UniqueItems',
        'Firstqty', 'RMQty', 'EndQty', 'Qty_Gold', 'Qty_Add', 'Qty_H_Add'
    ]

    existing_columns = [col for col in preferred_order if col in grouped_df.columns]
    remaining_columns = [col for col in grouped_df.columns if col not in existing_columns]
    final_columns = existing_columns + remaining_columns

    grouped_df = grouped_df[final_columns]
    return grouped_df


def prepare_export_data(df):
    """Prepare data for export by removing specified columns"""
    if df is None or df.empty:
        return df

    # Columns to exclude from export
    exclude_columns = ['WorkSheetId', 'StartTime', 'EndTime', 'JobId', 'WorkCenterName', 'PostingStatus']

    # Create a copy and remove excluded columns
    export_df = df.copy()
    for col in exclude_columns:
        if col in export_df.columns:
            export_df = export_df.drop(columns=[col])

    return export_df


def main():
    """Main Streamlit application"""

    # Title and description
    st.title("📊 Work Sheet Report with Advanced Search")
    st.markdown("---")

    # Get filter options
    job_orders, work_centers, categories, metals = get_filter_options()

    # Sidebar for filters
    with st.sidebar:
        st.header("🔍 Report Parameters")

        # Date inputs
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input(
                "From Date",
                value=date(2025, 7, 1),
                help="Select start date"
            )
        with col2:
            to_date = st.date_input(
                "To Date",
                value=date(2025, 7, 31),
                help="Select end date"
            )

        st.markdown("### Search Filters")

        # Work sheet reference
        work_ref = st.text_input(
            "Work Sheet Reference",
            value="",
            help="Leave empty to get all worksheets"
        )

        # Job Order Reference
        job_order_ref = st.selectbox(
            "Job Order Reference",
            options=[""] + job_orders,
            index=0,
            help="Select specific job order or leave empty for all"
        )

        # Work Center Name
        work_center_name = st.selectbox(
            "Work Center Name",
            options=[""] + work_centers,
            index=0,
            help="Select specific work center or leave empty for all"
        )

        # Item Category
        item_category = st.selectbox(
            "Item Category",
            options=[""] + categories,
            index=0,
            help="Select specific item category or leave empty for all"
        )

        # Metal
        metal = st.selectbox(
            "Metal Type",
            options=[""] + metals,
            index=0,
            help="Select specific metal type or leave empty for all"
        )

        # Grouping option
        group_by_center = st.checkbox("Group by Work Center", value=True)

        # Generate report button
        generate_report = st.button("🔄 Generate Report", type="primary", use_container_width=True)

        # Clear filters button
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            st.rerun()

    # Main content
    if generate_report:
        with st.spinner("Fetching data from database..."):
            df = fetch_data(from_date, to_date, work_ref, job_order_ref, work_center_name, item_category, metal)

        if df is not None and not df.empty:
            # Store in session state
            st.session_state.report_data = df
            st.session_state.group_by_center = group_by_center
            st.session_state.from_date = from_date
            st.session_state.to_date = to_date

            # Clear any existing PDF data when new report is generated
            if 'pdf_data' in st.session_state:
                del st.session_state.pdf_data
            if 'excel_data' in st.session_state:
                del st.session_state.excel_data

    # Display report if data exists in session state
    if 'report_data' in st.session_state:
        df = st.session_state.report_data
        group_by_center = st.session_state.group_by_center
        from_date = st.session_state.from_date
        to_date = st.session_state.to_date

        # Show applied filters
        st.info(f"📊 Applied Filters: Date: {from_date} to {to_date} | "
                f"Work Ref: {work_ref or 'All'} | "
                f"Job Order: {job_order_ref or 'All'} | "
                f"Work Center: {work_center_name or 'All'} | "
                f"Category: {item_category or 'All'} | "
                f"Metal: {metal or 'All'}")

        # Prepare display data
        if group_by_center:
            df_display = group_by_workcenter(df)
            df_display = prepare_export_data(df_display)
            st.success(f"✅ Found {len(df)} records grouped into {len(df_display)} work centers")
        else:
            df_display = prepare_export_data(df)
            st.success(f"✅ Found {len(df)} individual records")

        # Display summary metrics
        if not group_by_center:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                unique_worksheets = df['WorkSeetRef'].nunique()
                st.metric("Unique Worksheets", unique_worksheets)
            with col3:
                total_gold = df['Qty_Gold'].sum()
                st.metric("Total Gold Qty", f"{total_gold:.3f}")
            with col4:
                g21_add_qty = df[df['Metal'] == 'G21']['Qty_Add'].sum() if 'Metal' in df.columns else 0
                st.metric("G21 Add Qty", f"{g21_add_qty:.3f}")

        # Display data table
        st.subheader("📋 Report Data")

        if group_by_center:
            # Display grouped data
            work_centers = df['WorkCenterName'].unique()

            for work_center in sorted(work_centers):
                with st.expander(f"🏭 {work_center}", expanded=True):
                    center_data = df[df['WorkCenterName'] == work_center].copy()
                    center_display = prepare_export_data(center_data)

                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Worksheets", len(center_data))
                    with col2:
                        st.metric("Unique Items", center_data['ItemRef'].nunique())
                    with col3:
                        st.metric("Total Gold", f"{center_data['Qty_Gold'].sum():.3f}")
                    with col4:
                        st.metric("Total Add", f"{center_data['Qty_Add'].sum():.3f}")

                    st.dataframe(center_display, use_container_width=True)
                    st.markdown("---")
        else:
            st.dataframe(df_display, use_container_width=True, height=400)

        # Export section
        st.markdown("### 📤 Export Options")

        # Prepare exports section with buttons to generate files
        col_prep1, col_prep2 = st.columns(2)

        with col_prep1:
            if st.button("🔄 Prepare Excel File", use_container_width=True, help="Generate Excel file for download"):
                with st.spinner("Generating Excel file..."):
                    excel_data = create_excel_download(df, grouped=group_by_center)
                    st.session_state.excel_data = excel_data
                    st.success("✅ Excel file ready for download!")

        with col_prep2:
            if st.button("🔄 Prepare PDF File", use_container_width=True, help="Generate PDF file for download"):
                with st.spinner("Generating PDF file..."):
                    current_user = "System User"  # Replace with actual username if available
                    pdf_data = create_pdf_with_arabic_support(
                        df,
                        grouped=group_by_center,
                        username=current_user
                    )
                    st.session_state.pdf_data = pdf_data
                    st.success("✅ PDF file ready for download!")

        # Download buttons section
        st.markdown("---")
        export_col1, export_col2 = st.columns(2)

        # Excel Download


        # PDF Download
        with export_col2:
            if 'pdf_data' in st.session_state:
                filename = f"{'grouped_' if group_by_center else ''}worksheet_report_{from_date}_{to_date}.pdf"
                st.download_button(
                    label="📄 Download PDF File",
                    data=st.session_state.pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("📝 Click 'Prepare PDF File' button first")

        # Show file sizes if available
        if 'excel_data' in st.session_state or 'pdf_data' in st.session_state:
            st.markdown("##### 📊 File Information")
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                if 'excel_data' in st.session_state:
                    excel_size = len(st.session_state.excel_data) / 1024  # KB
                    st.info(f"Excel file size: {excel_size:.1f} KB")

            with info_col2:
                if 'pdf_data' in st.session_state:
                    pdf_size = len(st.session_state.pdf_data) / 1024  # KB
                    st.info(f"PDF file size: {pdf_size:.1f} KB")

        # Additional analysis for non-grouped data
        if not group_by_center and 'Metal' in df.columns:
            with st.expander("📊 Detailed Analysis"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Metal Distribution")
                    metal_summary = df.groupby('Metal').agg({
                        'Qty_Gold': 'sum',
                        'Qty_Add': 'sum',
                        'Qty_H_Add': 'sum',
                        'WorkSeetRef': 'count'
                    }).round(3)
                    metal_summary.columns = ['Gold Qty', 'Add Qty', 'H Add Qty', 'Count']
                    st.dataframe(metal_summary)

                with col2:
                    st.subheader("Work Center Summary")
                    workcenter_summary = df.groupby('WorkCenterName').agg({
                        'Qty_Gold': 'sum',
                        'Qty_Add': 'sum',
                        'Qty_H_Add': 'sum',
                        'WorkSeetRef': 'count'
                    }).round(3)
                    workcenter_summary.columns = ['Gold Qty', 'Add Qty', 'H Add Qty', 'Count']
                    st.dataframe(workcenter_summary)

                # Category Analysis if available
                if 'ItemCategory' in df.columns:
                    st.subheader("Item Category Analysis")
                    category_summary = df.groupby('ItemCategory').agg({
                        'Qty_Gold': 'sum',
                        'Qty_Add': 'sum',
                        'Qty_H_Add': 'sum',
                        'WorkSeetRef': 'count'
                    }).round(3)
                    category_summary.columns = ['Gold Qty', 'Add Qty', 'H Add Qty', 'Count']
                    st.dataframe(category_summary)

                # Job Order Analysis if available
                if 'JobOrderRef' in df.columns:
                    st.subheader("Top 10 Job Orders by Gold Usage")
                    job_summary = df.groupby('JobOrderRef').agg({
                        'Qty_Gold': 'sum',
                        'Qty_Add': 'sum',
                        'WorkSeetRef': 'count'
                    }).round(3).sort_values('Qty_Gold', ascending=False).head(10)
                    job_summary.columns = ['Gold Qty', 'Add Qty', 'Worksheet Count']
                    st.dataframe(job_summary)

    else:
        # Instructions when no data is loaded
        st.info("👈 Please use the sidebar to set your parameters and generate the report.")

        # Show available filter options
        with st.expander("📋 Available Filter Options Preview"):
            if job_orders:
                st.write("**Available Job Orders:**", len(job_orders), "options")
            if work_centers:
                st.write("**Available Work Centers:**", len(work_centers), "options")
            if categories:
                st.write("**Available Item Categories:**", len(categories), "options")
            st.write("**Available Metals:** G18, G21")


# إضافة cache للـ PDF generation
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_cached_pdf(df_dict, grouped, username, execution_datetime):
    """Generate PDF with caching to improve performance"""
    import pandas as pd

    # Convert dict back to DataFrame
    df = pd.DataFrame(df_dict)

    return create_pdf_with_arabic_support(
        df,
        grouped=grouped,
        username=username,
        execution_datetime=execution_datetime
    )


if __name__ == "__main__":
    main()



