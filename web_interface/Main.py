import streamlit as st
import folium
import numpy as np
import os
import socket
import pandas as pd
import pickle
import aiohttp
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack.errors import SlackApiError
from streamlit_folium import st_folium
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import bigquery
from st_on_hover_tabs import on_hover_tabs

st.set_page_config(
    layout="wide",
    page_title="Datavision",
    page_icon=":rocket:",
    initial_sidebar_state="expanded"
)

# Load environment variables from the .env file
load_dotenv()
# Access the secret key using the environment variable
SLACK_API_TOKEN = os.getenv('SLACK_API_TOKEN')
SLACK_API_TOKEN_BOT = os.getenv('SLACK_API_TOKEN_BOT')
SLACK_CHANNEL_ID = os.getenv('SLACK_CHANNEL_ID')

def send_slack_notification():
    
    # Get the hostname of the machine
    host_name = socket.gethostname()

    # Get the IP address associated with the hostname
    host_ip = socket.gethostbyname(host_name)

    client = WebClient(token=SLACK_API_TOKEN_BOT)
    message = f'Consulta generada desde IP: {host_ip}'

    try:
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=message
        )
        print(f"Message sent: {response['ts']}")
    except SlackApiError as e:
        print(f"Error sending message", e)



def KNN_model_predict(params, slider_count):


    
    # key_file_path = '/final_app/GCP_service_account/key_storage_ML.json'
    key_file_path = './GCP_service_account/key_storage_ML.json'
    bucket_name= 'data_clear_ml'
    file_name = 'modelo1_knn.pkl'

    # Set up authentication with service account key file


    storage_client = storage.Client.from_service_account_json(key_file_path)

    # Download the file from the bucket
    try:
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(file_name)

        model_data  = blob.download_as_bytes()

        knn_model_file = pickle.loads(model_data)

        indexes = knn_model_file.kneighbors([params],slider_count)[1][0]


    except Exception as e:
        print(f"Error downloading file: {e}")

    return indexes

def get_business_locations(indexes):


            # Also, in account service, you need to create the following permissions
            # BigQuery Data Viewer
            # BigQuery User

            keyfile_path = 'GCP_service_account/key_storage_ML.json'
           
            # Load credentials from the service account key file
           
            credentials = service_account.Credentials.from_service_account_file(
                keyfile_path,
            )

            # Create a BigQuery client using the obtained credentials
            client = bigquery.Client(credentials=credentials)

            print(indexes)

            query = f"""
                        SELECT
                                name,
                                latitude,
                                longitude
                        FROM
                                `bionic-store-413117.google.business_ml`
                        WHERE
                                index IN {indexes}
                        """
            # Run the query
            query_job = client.query(query)
            # Fetch results
            results = query_job.result()
            # Process and print results

            return results.to_dataframe()

def create_blank_map():
    # center on , add marker
    m = folium.Map(location=[38, -97], zoom_start=6)
    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=400)

    return st_data

def render_ML_graphic():
        
        data = {'lat': [-33, -33.1, -33.2], 'lon': [-58.8, -58.6, -58.7], 'size': [1000,2000,3000]}
        chart_data = pd.DataFrame(data)
        st.map(chart_data, size='size', zoom=8, use_container_width=True)

def render_ML_data():
                    pass

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'Dashboard', 'ML'], 
                         iconName=['home', 'dashboard', 'search'], default_choice=0)


if tabs=="Home":

            st.header("Datavision Solutions \U0001F3E0", divider='rainbow',anchor=False)


            st.subheader('Bienvenidos:')

            intro = '''    Somos *Datavision Solutions*, una consultora encargada de ofrecer soluciones avanzadas basadas en datos y les damos la bienvenida a nuestra plataforma en línea, presentando un innovador modelo de predicción, resultado de un minucioso análisis de mercado enfocado en la industria de comida rápida en los Estados Unidos.
            Este modelo ha sido desarrollado con la finalidad de proporcionar recomendaciones fundamentadas para aquellos emprendedores interesados en incursionar en el sector de comida rápida según la proximidad geográfica, la evaluación de usuarios, y la variedad de servicios ofrecidos por los establecimientos.
            Además tendrán la oportunidad de sumergirse en un nuestro dashboard, el cual proporciona un análisis visual profundo de los negocios de comida rápida y su posición en el dinámico mercado estadounidense junto con los Kpis propuestos ofreciendo insights valiosos sobre la interacción y la capacidad de adaptación de cada emprendimiento   
            '''
            st.markdown(intro)

            st.subheader('Autores:')
            authors = '''   **Belen Viglioglia Becker** - Data Analyst  
                            **Marcelo Ortiz** - Data Analyst   
                            **Mariano Popov** - Data Engineer  
                            **Alejandro Ramirez** - Data Engineer  
                            **Martin Riveros** - Data Scientist 
        '''
            st.markdown(authors)

elif tabs=="Dashboard":

        st.header("Dashboard \U0001F4CA", divider='rainbow',anchor=False)
        embed_url = "https://app.powerbi.com/view?r=eyJrIjoiMGJkZDkxNjUtOGQxNy00ZWVjLWIxM2YtZTdhNjdkOGNlMzg3IiwidCI6Ijk5ZTFlNzIxLTcxODQtNDk4ZS04YWZmLWIyYWQ0ZTUzYzFjMiIsImMiOjR9&pageName=ReportSection"
        st.components.v1.iframe(src=embed_url, height=700, width=1150)

elif tabs=="ML":

        st.header("Machine Learning Model \U0001F50E", divider='rainbow')
        
        col1, col2, col3= st.columns([1,1,2])
        with st.form("Trained_model"):
            
            with col1:           

                    st.header("Parametros")
                    slider_count  = st.slider("Cant a devolver", min_value=5, max_value=15)
                    toggle_takeout = st.toggle('Takeout')
                    toggle_delivery = st.toggle('Delivery')
                    toggle_kids = st.toggle('Entretenimientos de ninos')
                    toggle_creditcard = st.toggle('Acepta Tarjeta de Credito')
                    toggle_reservation = st.toggle('Se hacen reservas')
                    toggle_wifi = st.toggle('Wifi disponible')
            
            with col2:
                    
                    st.header('\U0001F30D')
                    toggle_dogs = st.toggle('Pet friendly')
                    toggle_alcohol = st.toggle('Venta de Alcohol')
                    toggle_hamburger= st.toggle('Hacen hamburguesas')
                    toggle_sandwich = st.toggle('Hacen sandwiches')
                    toggle_breakfast = st.toggle('Tienen desayuno')
                    toggle_ice = st.toggle('Tienen helados')
                    toggle_chiken = st.toggle('Tienen pollo')
                    toggle_mexican = st.toggle('Comida mexicana')
                    toggle_american = st.toggle('Comida yankee')      
            
            with col3:
                st.header('Hace doble click en el mapa')
                default_lat, default_lon = (38, -97)

                # Initialize session state variables
                if 'lat' not in st.session_state:
                    st.session_state.lat = default_lat

                if 'lon' not in st.session_state:
                    st.session_state.lon = default_lon

                m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=4)
                folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
                
                st_data = st_folium(m, width=600, height=350)
                                
                try:
                        st.session_state.lat = st_data['last_clicked']['lat']
                        st.session_state.lon = st_data['last_clicked']['lng']
                except:
                       print('mistake')
                
                lc_lat = st.session_state.lat
                lc_long = st.session_state.lon

            submitted = st.form_submit_button(label="Enviar",
                                                                    help=None,
                                                                    on_click=None,
                                                                    args=None,
                                                                    kwargs=None,
                                                                    type="primary",
                                                                    disabled=False,
                                                                    use_container_width=True)
                            
            if submitted:
                        
                        send_slack_notification()
                        
                        if st.session_state.keys()=='':
                                st.warning('Especifique un punto de interes en el plano', icon="⚠️")
                       
                        else:
                        
                            params = [lc_lat,
                                    lc_long,
                                    toggle_takeout,
                                    toggle_delivery,
                                    toggle_kids,
                                    toggle_creditcard,
                                    toggle_reservation,
                                    toggle_wifi,
                                    toggle_dogs,
                                    toggle_alcohol,
                                    toggle_hamburger,
                                    toggle_sandwich,
                                    toggle_breakfast,
                                    toggle_ice,
                                    toggle_chiken,
                                    toggle_mexican,
                                    toggle_american
                                ]

                            initial_text = "Ejecutando busqueda GCP Storage - modelo KNN"
                            middle_text = "Buscando indices en GCP BigQuery"
                            my_bar = st.progress(0, text=initial_text)

                            indexes = tuple(KNN_model_predict(params, slider_count))

                            
                            my_bar.progress(50, text=middle_text)
                            locations_df = get_business_locations(indexes)
                                
                            # calculate locations bounds

                            sw = [np.min(locations_df['latitude']), np.max(locations_df['longitude'])]
                            ne = [np.max(locations_df['latitude']), np.min(locations_df['longitude'])]

                                
                            c1, c2 = st.columns([1,1])
                            
                            with c1:        
                                    
                                    m1 = folium.Map(location=[np.mean(locations_df['latitude']), np.mean(locations_df['longitude'])], zoom_start=4)
                                                    
                                    for index, row in locations_df.iterrows():
                                                            
                                            folium.Marker(
                                                location=[row['latitude'], row['longitude']],
                                                popup=row['name']
                                                ).add_to(m1)
                                            
                                    # pass bunds to the map from returned data

                                    folium.Marker([st.session_state.lat, st.session_state.lon],popup="Ubicacion Seleccionada", tooltip='Ubicacion Seleccionada',icon=folium.Icon(color='red')).add_to(m1)


                                    m1.fit_bounds([sw, ne])
                                    st_folium(m1, height=400, width=700)
                            with c2:
                                    
                                    st.dataframe(locations_df)
                                    my_bar.empty()
                