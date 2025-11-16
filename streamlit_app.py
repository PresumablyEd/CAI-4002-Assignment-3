import streamlit as st
import pandas as pd

import backend  # local file: backend.py


st.set_page_config(
    page_title="Supermarket Association Mining",
    page_icon="🛒",
    layout="wide",
)


def get_transactions_df():
    """
    Helper that returns the current transactions DataFrame
    from session_state, or loads the default dataset.
    """
    if "transactions_df" not in st.session_state:
        try:
            st.session_state.transactions_df = backend.load_default_transactions()
        except FileNotFoundError:
            st.session_state.transactions_df = pd.DataFrame()
    return st.session_state.transactions_df


def get_products_df():
    """
    Helper that returns the current products DataFrame
    from session_state, or loads the default dataset.
    """
    if "products_df" not in st.session_state:
        try:
            st.session_state.products_df = backend.load_default_products()
        except FileNotFoundError:
            st.session_state.products_df = pd.DataFrame()
    return st.session_state.products_df


def home_page():
    st.title("🛒 Interactive Supermarket + Association Rule Mining")
    st.write(
    )

    st.markdown("### How to use this app")
    st.markdown(
        """
        1. Use the **Data Explorer** page to load and inspect the transactions dataset.  
        2. Use the **Shopping Simulator** page to build a fake shopping basket from the product list.  
        3. Use the **Association Rules** page to run Apriori / Eclat / FP-Growth   
        """
    )

    st.markdown("---")
    st.markdown("**Current datasets (from CSV):**")

    transactions_df = get_transactions_df()
    products_df = get_products_df()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transactions")
        if not transactions_df.empty:
            st.write(f"Rows: {len(transactions_df)}")
            st.dataframe(transactions_df.head())
        else:
            st.info("No `sample_transactions.csv` found or it could not be loaded.")

    with col2:
        st.subheader("Products")
        if not products_df.empty:
            st.write(f"Rows: {len(products_df)}")
            st.dataframe(products_df.head())
        else:
            st.info("No `products.csv` found or it could not be loaded.")


def data_explorer_page():
    st.title("📊 Data Explorer")

    st.markdown("#### Load transactions")
    use_default = st.checkbox("Use sample_transactions.csv from project folder", value=True)

    if use_default:
        df = get_transactions_df()
    else:
        uploaded = st.file_uploader("Upload your own transactions CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state.transactions_df = df
        else:
            df = get_transactions_df()

    if df.empty:
        st.warning("No transactions data available.")
        return

    st.success(f"Loaded {len(df)} transactions.")
    st.dataframe(df)

    st.markdown("#### Basic info")
    st.write(df.describe(include="all"))


def shopping_simulator_page():
    st.title("🧺 Shopping Simulator")

    products_df = get_products_df()
    if products_df.empty:
        st.warning("No products data available. Make sure `products.csv` is in the project folder.")
        return

    # ensure basket exists in session_state
    if "basket" not in st.session_state:
        st.session_state.basket = []

    st.markdown("Select items to add to the current shopping basket.")

    # Use a multiselect over product names
    all_products = products_df["product_name"].tolist()
    selected_products = st.multiselect(
        "Available items",
        options=all_products,
        default=[],
    )

    if st.button("Add selected items to basket"):
        # Extend basket with new unique items
        for item in selected_products:
            if item not in st.session_state.basket:
                st.session_state.basket.append(item)

    st.markdown("#### Current basket")
    if st.session_state.basket:
        # show each item with a delete button
        for idx, item in enumerate(st.session_state.basket):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(item)
            with col2:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.basket.pop(idx)
                    st.experimental_rerun()

        # button to clear the whole basket
        if st.button("Clear basket"):
            st.session_state.basket = []
            st.experimental_rerun()
    else:
        st.info("Basket is empty. Select items and click the button above.")

    st.markdown("---")
    st.markdown("#### Recommendations")

    if st.session_state.basket:
        # Placeholder call to backend (will return empty list until implemented)
        rules = backend.recommend_from_basket(st.session_state.basket)
        if rules:
            st.write("Recommended items based on your basket:")
            st.write(rules)
        else:
            st.info("No recommendations to show for this basket.")
    else:
        st.info("Add items to your basket to see recommendations.")


def association_rules_page():
    st.title("📈 Association Rule Mining")

    df = get_transactions_df()
    if df.empty:
        st.warning("No transactions loaded. Go to **Data Explorer** first or add `sample_transactions.csv`.")
        return

    st.markdown("Configure the mining parameters and choose an algorithm.")
    algorithm = st.selectbox("Algorithm", ["Apriori", "Eclat", "FP-Growth"])

    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.number_input("Min support", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with col2:
        min_confidence = st.number_input("Min confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    with col3:
        min_lift = st.number_input("Min lift", min_value=0.0, value=1.0, step=0.1)

    if st.button("Run mining"):
        with st.spinner("Mining rules..."):
            rules_df = backend.mine_association_rules(
                transactions_df=df,
                algorithm=algorithm.lower(),  # "apriori", "eclat", "fp-growth"
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
            )

        if isinstance(rules_df, pd.DataFrame) and not rules_df.empty:
            st.success(f"Found {len(rules_df)} rules.")
            st.dataframe(rules_df)
        else:
            st.info(
                "No rules returned yet. This is expected until the algorithms "
                "are implemented in `backend.py`."
            )


def main():
    # Simple single-file Streamlit app with sidebar navigation
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Explorer", "Shopping Simulator", "Association Rules"],
    )

    if page == "Home":
        home_page()
    elif page == "Data Explorer":
        data_explorer_page()
    elif page == "Shopping Simulator":
        shopping_simulator_page()
    elif page == "Association Rules":
        association_rules_page()


if __name__ == "__main__":
    main()
