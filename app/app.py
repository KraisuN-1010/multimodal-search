# app/app.py
import os
import streamlit as st
from PIL import Image
from engine import multimodal_search

# Force absolute layout limits for the UI grid
st.set_page_config(page_title="AI Multimodal E-commerce Search", layout="wide")

st.title("🛍️ Multimodal AI Product Search Engine")
st.subheader("Search your catalog seamlessly using text phrases or reference images")

# Sidebar - Controls & Asset Verification Status
st.sidebar.header("⚙️ System Status")
INDEX_PATH = "data/vector_index.bin"
MAPPING_PATH = "data/file_mapping.txt"

if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
    st.sidebar.success("FAISS Vector Index: Loaded & Active")
else:
    st.sidebar.error("FAISS Index Files Missing! Run engine.py first.")

top_k = st.sidebar.slider("Number of products to retrieve (Top K)", min_value=1, max_value=5, value=3)

# Main UI Tabs for different search types
tab1, tab2 = st.tabs(["💬 Text Search", "🖼️ Image Similarity Search"])

# --- TAB 1: TEXT TO IMAGE SEARCH ---
with tab1:
    st.markdown("### Type what you are looking for")
    text_query = st.text_input(
        label="Search Query Input", 
        placeholder="e.g., a clean formal shirt, something cozy for winter, running shoes...",
        label_visibility="collapsed"
    )
    
    if st.button("Search Products", key="text_search_btn"):
        if text_query.strip():
            with st.spinner("Scanning vector space..."):
                try:
                    # Run search against engine file tracking maps
                    results = multimodal_search(query_text=text_query, top_k=top_k)
                    
                    if not results:
                        st.info("No matching items found in catalog.")
                    else:
                        st.markdown("#### Top Matched Items:")
                        # Create dynamic responsive columns for our products
                        cols = st.columns(len(results))
                        for i, item in enumerate(results):
                            with cols[i]:
                                if os.path.exists(item["image_path"]):
                                    img = Image.open(item["image_path"])
                                    st.image(img, use_column_width=True)
                                    st.metric(
                                        label=f"Rank {i+1}", 
                                        value=os.path.basename(item["image_path"]), 
                                        delta=f"Match: {item['confidence_score']:.2%}"
                                    )
                                else:
                                    st.warning(f"File missing: {item['image_path']}")
                except Exception as e:
                    st.error(f"Search Engine Error: {str(e)}")
        else:
            st.warning("Please enter a valid search phrase.")

# --- TAB 2: IMAGE TO IMAGE SEARCH ---
with tab2:
    st.markdown("### Upload a reference product image to find similar items")
    uploaded_file = st.file_uploader("Drop an image file here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create a temporary local file pathway footprint so our engine can read it
        temp_dir = "data/temp_queries"
        os.makedirs(temp_dir, exist_ok=True)
        temp_img_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded image file payload to disk temporarily
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Display the user's uploaded input file target layout
        st.markdown("#### Your Uploaded Reference Item:")
        st.image(Image.open(temp_img_path), width=250)
        
        if st.button("Find Visually Similar Products", key="image_search_btn"):
            with st.spinner("Analyzing image features..."):
                try:
                    results = multimodal_search(query_image_path=temp_img_path, top_k=top_k)
                    
                    if not results:
                        st.info("No visually matching items found.")
                    else:
                        st.markdown("---")
                        st.markdown("#### Recommended Visual Alternatives:")
                        cols = st.columns(len(results))
                        for i, item in enumerate(results):
                            with cols[i]:
                                if os.path.exists(item["image_path"]):
                                    img = Image.open(item["image_path"])
                                    st.image(img, use_column_width=True)
                                    st.metric(
                                        label=f"Match Rank {i+1}", 
                                        value=os.path.basename(item["image_path"]), 
                                        delta=f"Similarity: {item['confidence_score']:.2%}"
                                    )
                except Exception as e:
                    st.error(f"Visual Search Engine Error: {str(e)}")