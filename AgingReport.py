import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from collections import deque
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus


# ----------------- Database Configuration -----------------
def create_db_engine():
    """Create a database connection engine."""
    try:
        server = "neg_data_server.arguserp.net"
        database = "R1029"
        username = "sa"
        password = "Argus@NEG"
        driver = "ODBC Driver 17 for SQL Server"

        connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        encoded_connection = quote_plus(connection_string)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_connection}")

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine

    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None


@st.cache_data(ttl=600)
def fetch_data(query, params=None):
    """Fetch data from the database."""
    try:
        engine = create_db_engine()
        if engine:
            return pd.read_sql(text(query), engine, params=params)
        return pd.DataFrame()
    except SQLAlchemyError as e:
        st.error(f"⚠️ Query Error: {str(e)}")
        return pd.DataFrame()


# ----------------- Business Logic -----------------
def convert_gold(row):
    """Convert gold quantities to 21K standard."""
    if row['currencyid'] == 2:
        return row['amount'] * 6 / 7  # 18K to 21K
    elif row['currencyid'] == 4:
        return row['amount'] * 8 / 7  # 24K to 21K
    return row['amount']


def process_fifo(debits, credits):
    """Process payments using FIFO principle"""
    debits_queue = deque(debits)
    payment_history = []

    for credit in sorted(credits, key=lambda x: x['date']):
        remaining_credit = credit['amount']

        while remaining_credit > 0 and debits_queue:
            current_debit = debits_queue[0]
            amount_to_apply = min(remaining_credit, current_debit['remaining'])
            current_debit['remaining'] -= amount_to_apply
            remaining_credit -= amount_to_apply

            if current_debit['remaining'] <= 0:
                current_debit['paid_date'] = credit['date']
                paid_debit = debits_queue.popleft()
                payment_history.append(paid_debit)

    payment_history.extend(debits_queue)
    return payment_history


def process_report(df):
    """Process and format aging reports"""
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.floor('D')
    df['paid_date'] = pd.to_datetime(df['paid_date'], errors='coerce').dt.floor('D')

    today = pd.Timestamp.today().floor('D')
    df['daydiff'] = np.where(
        df['paid_date'].isna(),
        (today - df['date']).dt.days,
        (df['paid_date'] - df['date']).dt.days
    )

    df['remaining'] = df['remaining'].fillna(0)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['paid_date'] = df['paid_date'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isna(x) else 'Unpaid')

    return df


def calculate_aging_reports_with_opening_balances(transactions, opening_balances):
    """Calculate aging reports including opening balances"""
    transactions['converted'] = transactions.apply(convert_gold, axis=1)
    transactions['abs_amount'] = transactions['amount'].abs()

    opening_cash = opening_balances[opening_balances['currencyid'] == 1]
    opening_gold = opening_balances[opening_balances['currencyid'] != 1]

    cash_debits, cash_credits = [], []
    gold_debits, gold_credits = [], []

    if not opening_balances.empty:
        fiscal_year = opening_balances['fiscalYear'].iloc[0]
        opening_date = datetime(fiscal_year - 1, 12, 31)

        for _, row in opening_cash.iterrows():
            if row['amount'] > 0:
                cash_debits.append({
                    'date': opening_date,
                    'reference': 'Opening Balance',
                    'amount': abs(row['amount']),
                    'remaining': abs(row['amount']),
                    'paid_date': None
                })
            else:
                cash_credits.append({'date': opening_date, 'amount': abs(row['amount'])})

        for _, row in opening_gold.iterrows():
            converted_gold = convert_gold(row)
            if row['amount'] > 0:
                gold_debits.append({
                    'date': opening_date,
                    'reference': 'Opening Balance',
                    'amount': abs(converted_gold),
                    'remaining': abs(converted_gold),
                    'paid_date': None
                })
            else:
                gold_credits.append({'date': opening_date, 'amount': abs(converted_gold)})

    for _, row in transactions.iterrows():
        converted_value = row['converted']
        if row['amount'] > 0:
            (gold_debits if row['currencyid'] != 1 else cash_debits).append({
                'date': row['date'],
                'reference': row['reference'],
                'amount': abs(converted_value),
                'remaining': abs(converted_value),
                'paid_date': None
            })
        else:
            (gold_credits if row['currencyid'] != 1 else cash_credits).append({
                'date': row['date'],
                'amount': abs(converted_value)
            })

    cash_results = process_fifo(sorted(cash_debits, key=lambda x: x['date']), cash_credits)
    gold_results = process_fifo(sorted(gold_debits, key=lambda x: x['date']), gold_credits)

    return process_report(pd.DataFrame(cash_results)), process_report(pd.DataFrame(gold_results))

# ----------------- User Interface -----------------
def main():
    st.set_page_config(page_title="Invoice Aging System", layout="wide")
    st.title("📊 Invoice Aging Report - Cash and Gold")

    groups = fetch_data("SELECT recordid, name FROM figrp ORDER BY name")
    group_names = ["Select Group..."] + groups['name'].tolist()
    selected_group = st.sidebar.selectbox("Group", group_names)

    customers = pd.DataFrame()
    if selected_group != "Select Group...":
        group_id = int(groups[groups['name'] == selected_group]['recordid'].values[0])
        customers = fetch_data(
            "SELECT recordid, name, reference FROM fiacc WHERE groupid = :group_id",
            {"group_id": group_id}
        )

    customer_list = ["Select Customer..."] + [f"{row['name']} ({row['reference']})" for _, row in customers.iterrows()]
    selected_customer = st.sidebar.selectbox("Customer", customer_list)

    start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    fiscal_year = st.sidebar.number_input("📅 السنة المالية", min_value=2020, max_value=datetime.now().year,
                                          value=datetime.now().year, step=1)
    discount_percent = st.sidebar.number_input(
        "Discount Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        format="%.1f"
    )

    if st.sidebar.button("Generate Report"):
        if selected_customer == "Select Customer...":
            st.error("Please select a customer.")
            return

        customer_id = int(customers.iloc[customer_list.index(selected_customer) - 1]['recordid'])

        transactions = fetch_data(
            """SELECT date, amount, currencyid, reference 
            FROM fitrx 
            WHERE accountid = :acc_id 
                AND date BETWEEN :start AND :end
            ORDER BY date""",
            {
                "acc_id": customer_id,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            }
        )
        if discount_percent > 0:
            cash_debits_mask = (transactions['currencyid'] == 1) & (transactions['amount'] > 0)
            transactions.loc[cash_debits_mask, 'amount'] *= (1 - discount_percent / 100)
        opening_balances = fetch_data(
            "SELECT amount, currencyid, fiscalYear FROM fioba WHERE accountid = :acc_id AND fiscalYear = :fiscal_year",
            {"acc_id": customer_id, "fiscal_year": fiscal_year}
        )

        if transactions.empty and opening_balances.empty:
            st.warning("No transactions found.")
            return

        transactions['date'] = pd.to_datetime(transactions['date'])
        cash_report, gold_report = calculate_aging_reports_with_opening_balances(transactions, opening_balances)

        if not cash_report.empty:
            st.subheader("Cash Report")
            st.dataframe(
                cash_report[['date', 'reference', 'amount','remaining', 'paid_date', 'daydiff']],
                column_config={
                    "date": "Invoice Date",
                    "reference": "Reference",
                    "amount": "Amount",
                    "paid_date": "Payment Date",
                    "remaining": "Remaining",
                    "daydiff": "Aging Days"
                },
                hide_index=True,
                use_container_width=True
            )
            csv_cash = cash_report.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="Download Cash Report",
                data=csv_cash,
                file_name=f"Cash_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No cash transactions.")

        if not gold_report.empty:
            st.subheader("Gold Report (21K Equivalent)")
            st.dataframe(
                gold_report[['date', 'reference', 'amount','remaining', 'paid_date', 'daydiff']],
                column_config={
                    "date": "Invoice Date",
                    "reference": "Reference",
                    "amount": "Gold G21",
                    "paid_date": "Payment Date",
                    "remaining":"Remaining",
                    "daydiff": "Aging Days"
                },
                hide_index=True,
                use_container_width=True
            )
            csv_gold = gold_report.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="Download Gold Report",
                data=csv_gold,
                file_name=f"Gold_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No gold transactions.")


if __name__ == "__main__":
    main()