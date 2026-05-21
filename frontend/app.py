import os
import streamlit as st
import requests
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Olist — Prédiction livraison", page_icon="📦")
st.title("📦 Prédiction du temps de livraison")
st.caption("Olist Brazilian E-Commerce — RandomForestRegressor")

st.sidebar.header("Paramètres de la commande")

price = st.sidebar.number_input("Prix total (BRL)", min_value=1.0, value=89.90,
step=10.0)
freight = st.sidebar.number_input("Frais de port (BRL)", min_value=0.0, value=12.50,
step=1.0)
installments = st.sidebar.slider("Nombre de versements", min_value=1, max_value=12,
value=1)
payment_value = st.sidebar.number_input("Montant payé (BRL)", min_value=1.0, value=102.40)
n_items = st.sidebar.slider("Nombre d'articles", min_value=1, max_value=10, value=1)
purchase_dt = st.sidebar.date_input("Date d'achat", value=datetime(2018, 6, 10))
seller_state = st.sidebar.selectbox("État vendeur",
["SP","RJ","MG","RS","PR","SC","BA","GO","ES","CE"])
customer_state = st.sidebar.selectbox("État client",
["SP","RJ","MG","RS","PR","SC","BA","GO","ES","CE"])

if st.button("🔮 Prédire le temps de livraison", type="primary"):
    payload = {
        "price": price,
        "freight_value": freight,
        "payment_installments": installments,
        "payment_value": payment_value,
        "order_item_id": n_items,
        "order_purchase_timestamp": datetime.combine(purchase_dt, datetime.min.time()).isoformat(),
        "seller_state": seller_state,
        "customer_state": customer_state,
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        days = resp.json()["delivery_time_days"]
        st.success(f"⏱ Temps de livraison prédit : **{days:.1f} jours**")
        if days < 7:
            st.info("Livraison rapide — client probablement satisfait.")
        elif days < 15:
            st.warning("Livraison dans la moyenne Olist (médiane : ~12 jours).")
        else:
            st.error("Livraison longue — risque d'insatisfaction élevé.")
    except requests.RequestException as e:
        st.error(f"Erreur API : {e}")
        
with st.expander("ℹ️ À propos du modèle"):
    st.markdown("""
    **Algorithme** : RandomForestRegressor (scikit-learn)
    **Entraîné sur** : ~96 000 commandes Olist livrées (2016-2018)
    **Features** : prix, fret, versements, états vendeur/client, jour/mois d'achat
    **Métrique** : R² > 0.70 | MAE < 5 jours
    **Artefact** : stocké dans MinIO, chargé au démarrage de l'API
    """)