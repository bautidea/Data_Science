import streamlit as st
import streamlit.components.v1 as components
from objeto import function

class deploy():
    def __init__(self):
            
        st.set_page_config(layout='wide')
        
        # Instancio el modulo y almaceno los velores iniciales.
        if 'function' not in st.session_state:
            self.function = function()    

        # Asigno los valores por default a cada variable.
        if 'age_value' not in st.session_state:
            st.session_state.age_value = '' 
        
        if 'sex_value' not in st.session_state:
            st.session_state.sex_value = ''
        
        if 'dependant_value' not in st.session_state:
            st.session_state.dependant_value = ''
                    
        if 'education_value' not in st.session_state:
            st.session_state.education_value = ''
                                            
        if 'marital_value' not in st.session_state:
            st.session_state.marital_value = ''
        
        if 'income_value' not in st.session_state:
            st.session_state.income_value = ''
                    
        if 'card_value' not in st.session_state:
            st.session_state.card_value = ''
        
        if 'on_book_value' not in st.session_state:
            st.session_state.on_book_value = ''
        
        if 'relationship_value' not in st.session_state:
            st.session_state.relationship_value = ''
        
        if 'inactive_value' not in st.session_state:
            st.session_state.inactive_value = ''
                    
        if 'contacts_value' not in st.session_state:
            st.session_state.contacts_value = ''
                    
        if 'credit_value' not in st.session_state:
            st.session_state.credit_value = ''
                    
        if 'revolvin_value' not in st.session_state:
            st.session_state.revolvin_value = ''
                    
        if 'open_buy_value' not in st.session_state:
            st.session_state.open_buy_value = ''
        
        if 'amt_chng_value' not in st.session_state:
            st.session_state.amt_chng_value = ''
        
        if 'trans_amt_value' not in st.session_state:
            st.session_state.trans_amt_value = ''
        
        if 'trans_ct_value' not in st.session_state:
            st.session_state.trans_ct_value = ''
                    
        if 'ct_chng_value' not in st.session_state:
            st.session_state.ct_chng_value = ''
                    
        if 'utilization_value' not in st.session_state:
            st.session_state.utilization_value = ''                                                                                                                                                                                                  

    # Creo funcion para crear los elementos visuales.
    # La misma sera llamada desde fuera de la clase.
    def window(self):
        # Creo numer input para el numero de socio.
        self.id = st.number_input(
            'Ingrese numero de cliente',
            value=0,
            min_value=0                
        )
        
        # Creo conteiner para mostrar los resultados de la prediccion.
        self.container_result = st.container()
        
        # Creo columnas para centrar los textos.
        with self.container_result:
            self.col_container_izq, self.col_container_med, self.col_container_der = st.columns([1,3,1])
            
        # Si se ingresa algun valor al numer_input me ejecuta los siguientes metodos.  
        if self.id:
            try:
                self.show_user()
                self.show_prediction()
            except:
                # Escribo mensaje de error en el conteiner.
                with self.col_container_med:
                    st.title('Error al encontrar el cliente')
                # Reseteo a los valores por default.   
                st.session_state.age_value = ''
                st.session_state.sex_value = ''
                st.session_state.dependant_value = ''
                st.session_state.education_value = ''
                st.session_state.marital_value = ''
                st.session_state.income_value = ''
                st.session_state.card_value = ''
                st.session_state.on_book_value = ''
                st.session_state.relationship_value = ''
                st.session_state.inactive_value = ''
                st.session_state.contacts_value = ''
                st.session_state.credit_value = ''
                st.session_state.revolvin_value = ''
                st.session_state.open_buy_value = ''
                st.session_state.amt_chng_value = ''
                st.session_state.trans_amt_value = ''
                st.session_state.trans_ct_value = ''
                st.session_state.ct_chng_value = ''
                st.session_state.utilization_value = ''

        # Contenedor para los datos del cliente.
        self.display_info = st.expander('Informacion del cliente')
        with self.display_info:
            # Columnas para separar los datos
            self.col1_cont, self.col2_cont, self.col3_cont, self.col4_cont = st.columns(4)  
            
            with self.col1_cont:          
                self.age = st.text_input(
                    'Edad',
                    disabled=True,
                    value=st.session_state.age_value
                )
                
                self.gender = st.text_input(
                    'Sexo',
                    disabled=True,
                    value=st.session_state.sex_value
                )
                
                self.dependant = st.text_input(
                    'Dependientes',
                    disabled=True,
                    value= st.session_state.dependant_value
                )
                
                self.education = st.text_input(
                    'Educacion',
                    disabled=True,
                    value = st.session_state.education_value
                )

                self.marital = st.text_input(
                    'Estado civil',
                    disabled=True,
                    value= st.session_state.marital_value
                )

            with self.col2_cont:                
                self.income = st.text_input(
                    'Ingresos Anuales',
                    disabled=True,
                    value= st.session_state.income_value
                )

                self.card = st.text_input(
                    'Tarjeta',
                    disabled=True,
                    value= st.session_state.card_value
                )

                self.months_on_book = st.text_input(
                    'Meses en el banco',
                    disabled=True,
                    value= st.session_state.on_book_value
                ) 
                
                self.total_relationship = st.text_input(
                    'Numero de productos',
                    disabled=True,
                    value= st.session_state.relationship_value
                ) 
                
                self.months_inactive = st.text_input(
                    'Meses inactivos ultimos 12 meses',
                    disabled=True,
                    value= st.session_state.inactive_value
                )

            with self.col3_cont:                        
                self.contacts = st.text_input(
                    'Contactos ultimos 12 meses',
                    disabled=True,
                    value= st.session_state.contacts_value
                )                
                
                self.credit_limit = st.text_input(
                    'Limite de credito',
                    disabled=True,
                    value= st.session_state.credit_value
                )               
                
                self.revolvin_ball = st.text_input(
                    'Saldo rotatorio total',
                    disabled=True,
                    value= st.session_state.revolvin_value
                ) 
                                
                self.avg_open_buy = st.text_input(
                    'Comprar lineas de credito',
                    disabled=True,
                    value= st.session_state.open_buy_value
                )
                               
                self.amt_chng = st.text_input(
                    'Cambio en las transacciones',
                    disabled=True,
                    value= st.session_state.amt_chng_value
                )
            
            with self.col4_cont:                
                self.trans_amt = st.text_input(
                    'Monto toal en las transacciones',
                    disabled=True,
                    value= st.session_state.trans_amt_value
                )
                               
                self.trans_ct = st.text_input(
                    'Conteo de transacciones',
                    disabled=True,
                    value= st.session_state.trans_ct_value
                )
                
                self.ct_chng = st.text_input(
                    'Cambio en el conteo de transacciones',
                    disabled=True,
                    value= st.session_state.ct_chng_value
                )
                               
                self.utilization = st.text_input(
                    'Ratio de uso de la tarjeta',
                    disabled=True,
                    value= st.session_state.utilization_value
                )   
    # Funcion para que se muestren los datos del usuario:
    def show_user(self):
        # Ejecuto metodo del modulo 'objeto', para obtener el registro deseado.
        df = self.function.load_user(self.id)
        
        st.session_state.age_value = df[0]
        st.session_state.sex_value = df[1]
        st.session_state.dependant_value = df[2]
        st.session_state.education_value = df[3]
        st.session_state.marital_value = df[4]
        st.session_state.income_value = df[5]
        st.session_state.card_value = df[6]
        st.session_state.on_book_value = df[7]
        st.session_state.relationship_value = df[8]
        st.session_state.inactive_value = df[9]
        st.session_state.contacts_value = df[10]
        st.session_state.credit_value = df[11]
        st.session_state.revolvin_value = df[12]
        st.session_state.open_buy_value = df[13]
        st.session_state.amt_chng_value = df[14]
        st.session_state.trans_amt_value = df[15]
        st.session_state.trans_ct_value = df[16]  
        st.session_state.ct_chng_value = df[17]
        st.session_state.utilization_value = df[18]
    
    # Funcion para predecir si el cliente cargado se ira del banco.
    def show_prediction (self):
        # Ejecuto metodo del modulo 'objeto', para obtener html de los graficos 
        # de las predicciones.
        plot_user_html, plot_all_html, plot_waterfall, prediction = self.function.predict_user(self.id)
        
        with self.container_result:
            # Mefijo si mi prediccion es positiva o negativa.
            if prediction ==0:
                with self.col_container_med:
                    st.title('No hay indicios de que el cliente abandone el banco')
            
            else:
                with self.col_container_med:
                    st.title('El cliente podria abandonar el banco')
                    
            components.html(plot_user_html)
            
            # Creo columnas adentro del contenedor para plotear los otros dos graficos.
            display_col1, display_col2 = st.columns(2)
            
            with display_col1:
                components.html(plot_all_html, height=400)
                
            with display_col2:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(plot_waterfall)
    
    
if __name__ == '__main__':
    
    deploy_model = deploy()
    deploy_model.window()