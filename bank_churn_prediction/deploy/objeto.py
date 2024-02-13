import matplotlib
import pandas as pd
import streamlit as st
import pickle
import shap
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class function():
    
    # Al instanciar este modulo, se carga el dataframe donde se encuentran
    # todos los clientes.
    def __init__(self):
        # Cargo csv.
        st.session_state.data = pd.read_csv('../data/bank_churners_filtered.csv')
        # Renombro csv para que el mismo pueda ser llamado por fuera del metodo init.
        self.data = st.session_state.data
        # Elimino campos inecesarios.
        self.data.drop(columns=['Unnamed: 0', 'Attrition_Flag'], inplace=True)
        # Aumento el numero del indice en 100, para su mejor interpretabilidad.
        self.data.index = self.data.index + 100
        
        # Cargo el modelo desde pickle
        with open ('../models/bank_churn_final_model.pkl', 'rb') as f_model:
            st.session_state.model = pickle.load(f_model)
        self.model = st.session_state.model

    # Metodo que devuelve el registro del cliente deseado.
    def load_user (self, id):
        registro = self.data.loc[id, :].values.tolist()
        return registro
    
    # Metodo para predecir si un cliente se quedara o abandonara el banco.
    def predict_user (self, id):
        # Primero obtengo el orden de las columnas, para realizar la prediccion.
        # Aplico la misma transformacion a mis datos que se hizo en el analisis de modelos.
        ct = ColumnTransformer([
            ('dum', OneHotEncoder(drop='first'), make_column_selector(dtype_include=object)),
            ('scale', StandardScaler(), make_column_selector(dtype_exclude=object))
        ])

        X = pd.DataFrame(data=ct.fit_transform(self.data), index=self.data.index)

        # Obtengo el nombre de las columnas.
        cols = ct.transformers_[0][1].get_feature_names().tolist()
        for i in ct.transformers_[1][2]:
            cols.append(i)
        # Renombro columnas.
        X.columns = cols
        # Elimino columnas que no se usan en la prediccion.
        X.drop(columns='x4_Platinum', inplace=True)
        
        # Creo el explicador para calcular los valores de shap.
        explainer = shap.TreeExplainer(self.model['xgb'])
        
        # Explico los valores del cliente seleccionado.
        chosen_client = X.loc[[id]]
        shap_values_cliente = explainer.shap_values(chosen_client)
        plot_user_shap = shap.force_plot(explainer.expected_value, shap_values_cliente, chosen_client)
        plot_user_html = f"<head>{shap.getjs()}</head><body>{plot_user_shap.html()}</body>"
        
        # Explico los valores dentro de un intervalo del cliente seleccionado
        chosen_interval = X.loc[id-200:id+200,:]
        shap_values_all = explainer.shap_values(chosen_interval)
        plot_all_shap = shap.force_plot(explainer.expected_value, shap_values_all, chosen_interval)
        plot_all_html = f"<head>{shap.getjs()}</head><body>{plot_all_shap.html()}</body>"
        
        # Creo un grafico para mostrar mejor la variacion de la explicacion.
        explainer_waterfall = shap.Explainer(self.model['xgb'], X)
        shap_values_waterfall = explainer_waterfall(chosen_client)
        plot_waterfall = shap.plots.waterfall(shap_values_waterfall[0])
        
        # Predigo el resultado sobre si el cliente abandonara o no el banco.
        prediction = self.model['xgb'].predict(X.loc[[id]])[0]
        
        return plot_user_html, plot_all_html, plot_waterfall, prediction