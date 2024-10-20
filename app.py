import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from agent_flow.agent import Agent
import uuid
from dotenv import load_dotenv
from PIL import Image
import io
import base64

load_dotenv()

# Spotify API credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URL')

# Spotify scopes
SCOPE = 'user-read-recently-played user-library-read user-follow-read playlist-read-collaborative playlist-read-private playlist-modify-public playlist-modify-private'

# Initialize Spotify client
sp_oauth = SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, scope=SCOPE, cache_path=None)

def image_to_data_url(image_file):
    if image_file is None:
        return None
    
    file_extension = image_file.name.split('.')[-1].lower()
    mime_type = f"image/{file_extension}"
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    base64_image = base64.b64encode(image_file.getvalue()).decode()
    return f"data:{mime_type};base64,{base64_image}"

def main():
    st.set_page_config(page_title="Spotify Playlist Creator", page_icon="🎵")
    st.title("Spotify Playlist Creator")
    
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Initialize session state
    if 'token_info' not in st.session_state:
        st.session_state.token_info = None

    # Check if we're in the authentication callback
    if 'code' in st.query_params:
        code = st.query_params['code']
        st.session_state.token_info = sp_oauth.get_access_token(code)
        st.query_params.clear()
        st.rerun()

    # If we don't have a token, get it
    if not st.session_state.token_info:
        auth_url = sp_oauth.get_authorize_url()
        st.markdown(f'<a href="{auth_url}" target="_self" class="spotify-button">Login to Spotify</a>', unsafe_allow_html=True)
        st.stop()

    # If we have a token, create a Spotify client
    sp = spotipy.Spotify(auth=st.session_state.token_info['access_token'])
    
    try:
        # Get user info
        user = sp.current_user()
        st.write(f"Welcome, {user['display_name']}!")

        # Create the agent with the user's Spotify token and ID
        if 'agent' not in st.session_state:
            st.session_state.agent = Agent(st.session_state.token_info['access_token'], user['id'])

        # User input
        input_type = st.radio("Choose input type:", ("Image Upload", "Text Description"))

        user_input = None
        if input_type == "Image Upload":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            image_string = image_to_data_url(uploaded_file)
            if image_string:
                user_input = image_string
        else:
            user_input = st.text_area("Enter a description for your playlist:")

        num_songs = st.slider("Number of songs in the playlist:", 5, 50, 10)

        if st.button("Create Playlist"):
            if user_input:
                with st.spinner("Creating your playlist..."):

                    if input_type == "Image Upload":
                        question = f"Create a playlist with {num_songs} songs based on this image."
                        results = st.session_state.agent.process_request(question, image_string=user_input)
                    else:
                        question = f"Create a playlist with {num_songs} songs that match the following description: {user_input}"
                        results = st.session_state.agent.process_request(question)

                    # Display the final result (assuming the last event contains the playlist information)
                    if results:
                        st.success("Playlist created successfully!")
                        st.write(results[-1])
                    else:
                        st.error("Failed to create playlist. Please try again.")
            else:
                st.warning("Please provide an input before creating a playlist.")

    except spotipy.SpotifyException:
        # Token might be expired or invalid
        st.session_state.token_info = None
        st.error("Session expired. Please log in again.")
        st.rerun()

if __name__ == "__main__":
    main()