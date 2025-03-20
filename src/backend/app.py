import streamlit as st
import numpy as np
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import sys
from PIL import Image
import io
import base64
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ✅ Set page config FIRST before any other code
st.set_page_config(page_title="HomeSafe", layout="wide", page_icon="🏠")

# Now continue with other imports

# Add the parent directory to the path so we can import from src modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import model and processing functions
# These imports should work now that we've added the parent directory to the path
try:
    from model_loader import model
    from process_image import preprocess_image
    # Import the function directly from the module
    from edge_detection import perform_edge_detection
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all required modules are in the correct location.")

# Initialize session state variables if they don't exist yet
if 'has_analysis' not in st.session_state:
    st.session_state.has_analysis = False
if 'severity' not in st.session_state:
    st.session_state.severity = None
if 'severity_description' not in st.session_state:
    st.session_state.severity_description = None
if 'length' not in st.session_state:
    st.session_state.length = 0
if 'width' not in st.session_state:
    st.session_state.width = 0
if 'original_image_path' not in st.session_state:
    st.session_state.original_image_path = None
if 'edge_image_path' not in st.session_state:
    st.session_state.edge_image_path = None
if 'has_crack' not in st.session_state:
    st.session_state.has_crack = False

# Function to determine severity based on crack dimensions
def analyze_severity(length, width):
    """
    Analyze the severity of a crack based on its dimensions.
    Returns a severity level and description.
    """
    if length == 0 or width == 0:
        return "None", "No crack detected"

    # Simple severity classification based on length and width
    if length > 100 and width > 5:
        return "High", "Major structural concern requiring immediate professional inspection"
    elif length > 50 or width > 3:
        return "Medium", "Moderate concern requiring monitoring and potential repair"
    else:
        return "Low", "Minor crack, monitor for changes"

#Email Sending
def send_email_report(user_email, user_name, address, image_path, edge_image_path, severity, severity_description, length, width):
    """
    Send an email report to the user with the analysis results.
    """
    try:
        # ✅ Use environment variables for email credentials
        # sender_email = os.getenv("EMAIL_ADDRESS")
        # password = os.getenv("EMAIL_PASSWORD")
        sender_email = st.secrets["email"]["address"]
        password = st.secrets["email"]["password"]

        # Debug prints to verify the environment variables are loaded correctly
        #st.write(f"Using email: {sender_email}")
        
        if not sender_email or not password:
            st.error("Email credentials are missing. Please check your .env file.")
            return False

        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = f'HomeSafe AI Inspection Report - {datetime.now().strftime("%Y-%m-%d")}'
        msg['From'] = sender_email
        msg['To'] = user_email

        # Email body
        body = f"""
        <html>
        <body>
        <h2>HomeSafe AI Inspection Report</h2>
        <p>Dear {user_name},</p>
        <p>Thank you for using HomeSafe AI for your property inspection. Below are the results of our analysis:</p>

        <h3>Property Details:</h3>
        <p>Address: {address}</p>

        <h3>Crack Analysis Results:</h3>
        <p><strong>Severity Level:</strong> {severity}</p>
        <p><strong>Assessment:</strong> {severity_description}</p>
        <p><strong>Crack Length:</strong> {length:.2f} pixels</p>
        <p><strong>Crack Width:</strong> {width:.2f} pixels</p>

        <h3>Recommendations:</h3>
        <p>{get_recommendations(severity)}</p>

        <p>Best regards,<br>
        HomeSafe AI Team</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        # ✅ Attach the original image
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img = MIMEImage(img_data)
            img.add_header('Content-Disposition', 'attachment', filename='original_image.jpg')
            msg.attach(img)

        # ✅ Attach the edge detection image
        with open(edge_image_path, 'rb') as img_file:
            img_data = img_file.read()
            img = MIMEImage(img_data)
            img.add_header('Content-Disposition', 'attachment', filename='edge_detection.jpg')
            msg.attach(img)

        # ✅ Send email using SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)

        return True
    
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        st.error(f"Error details: {str(e)}")
        return False

# Function to get recommendations based on severity
def get_recommendations(severity):
    if severity == "High":
        return "We recommend immediate consultation with a structural engineer. This crack indicates a potential structural issue that should be addressed promptly."
    elif severity == "Medium":
        return "We recommend monitoring this crack over the next 3-6 months. If you notice any expansion or additional cracks forming, please consult a home inspector or structural engineer."
    else:  # Low
        return "This appears to be a superficial crack. We recommend monitoring it and sealing it with appropriate filler to prevent moisture ingress."

# Function to save temporary image files
def save_temp_image(image, filename):
    """Save image to a temporary file and return the path"""
    temp_dir = os.path.join(parent_dir, "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, filename)

    if isinstance(image, np.ndarray):
        cv2.imwrite(file_path, image)
    else:
        image.save(file_path)

    return file_path

# Function to perform edge detection (fallback if import fails)
def local_edge_detection(image):
    """
    Perform edge detection on the input image.
    This is a fallback function in case the import fails.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return edges

# Function to measure crack dimensions
def measure_crack_dimensions(edge_image):
    """
    Measure the dimensions of the crack from the edge-detected image.
    Returns (length, width) in pixels.
    """
    # Find contours
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_length = 0
    max_width = 0

    for contour in contours:
        # Calculate arc length (crack length)
        length = cv2.arcLength(contour, closed=False)
        total_length += length

        # Calculate bounding box (crack width)
        _, _, width, height = cv2.boundingRect(contour)
        max_width = max(max_width, max(width, height))

    return total_length, max_width

# Function to get image as base64 for email
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Preprocess image for model (fallback function if import fails)
def local_preprocess_image(image):
    """
    Preprocess the image for the model.
    This is a fallback function in case the import fails.
    """
    # Resize to the target size expected by the model
    target_size = (224, 224)
    if isinstance(image, Image.Image):
        image = image.resize(target_size)
        image = np.array(image)
    else:
        image = cv2.resize(image, target_size)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# Load the logo image
project_root = os.path.dirname(parent_dir)
logo_path = os.path.join(project_root, "assets", "logo.png")
try:
    logo_image = Image.open(logo_path)
except Exception as e:
    st.warning(f"Could not load logo: {e}")
    logo_image = None

# Streamlit UI
col1, col2 = st.columns([1, 3])

with col1:
    if logo_image is not None:
        st.image(logo_image, width=250)  # Display the logo
    else:
        st.write("HomeSafe")

with col2:
    st.title("HomeSafe: AI-Powered Housing Inspection")

st.write("Upload an image to detect and analyze cracks in your property.")

# Create two columns for form and results
col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("User Information")

    # User details form
    with st.form(key='user_details'):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        address = st.text_area("Property Address")

        # File uploader
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        submit_button = st.form_submit_button(label='Analyze Image')

# Display results in the second column
with col4:
    st.subheader("Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Original Image", "Edge Detection", "Report"])

    # Process the image if the button is clicked and an image is uploaded
    if submit_button and uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            with tab1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            # Save the original image
            original_image_path = save_temp_image(image, f"original_{uploaded_file.name}")
            st.session_state.original_image_path = original_image_path

            # Convert PIL Image to numpy array for OpenCV
            image_np = np.array(image)

            # Use imported functions if available, fallback to local implementations if not
            try:
                # Preprocess image for model
                processed_image = preprocess_image(image)
                # Predict if there's a crack using the model
                prediction = model.predict(processed_image)
                confidence = prediction[0][0]
            except NameError:
                st.warning("Model loading failed. Using fallback detection method.")
                # Fallback - assume there is a crack for analysis purposes
                confidence = 0.7

            # Determine if there's a crack
            has_crack = confidence > 0.5
            st.session_state.has_crack = has_crack

            # Process only if crack is detected
            if has_crack:
                # Try to use the imported edge detection function
                try:
                    edges = perform_edge_detection(original_image_path,
                                                os.path.join(parent_dir, "temp_images", f"edges_{uploaded_file.name}"))
                    # Read back the saved edge image
                    edge_image_path = os.path.join(parent_dir, "temp_images", f"edges_{uploaded_file.name}")
                    edges = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
                except (NameError, TypeError):
                    # Fallback to local edge detection
                    edges = local_edge_detection(image_np)
                    edge_image_path = save_temp_image(edges, f"edges_{uploaded_file.name}")

                st.session_state.edge_image_path = edge_image_path

                # Display edge detection image
                with tab2:
                    st.image(edges, caption="Edge Detection", use_column_width=True)

                # Measure crack dimensions
                length, width = measure_crack_dimensions(edges)
                st.session_state.length = length
                st.session_state.width = width

                # Analyze severity
                severity, severity_description = analyze_severity(length, width)
                st.session_state.severity = severity
                st.session_state.severity_description = severity_description

                # Mark that we have analysis results
                st.session_state.has_analysis = True
            else:
                with tab2:
                    st.info("No cracks detected in the image.")

                # Mark that we have analysis but no cracks
                st.session_state.has_analysis = True
                st.session_state.severity = "None"
                st.session_state.severity_description = "No cracks detected"
                st.session_state.length = 0
                st.session_state.width = 0

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please try again with a different image.")

    # Display the report tab content based on session state
    if st.session_state.has_analysis:
        with tab3:
            st.markdown(f"### Crack Analysis Report")
            
            if st.session_state.has_crack:
                st.markdown(f"**Severity Level:** {st.session_state.severity}")
                st.markdown(f"**Assessment:** {st.session_state.severity_description}")
                st.markdown(f"**Crack Length:** {st.session_state.length:.2f} pixels")
                st.markdown(f"**Crack Width:** {st.session_state.width:.2f} pixels")
                
                st.markdown("### Recommendations")
                st.markdown(get_recommendations(st.session_state.severity))
                
                # Send email report button
                if st.button("Send Report to Email"):
                    if send_email_report(email, name, address, 
                                      st.session_state.original_image_path, 
                                      st.session_state.edge_image_path,
                                      st.session_state.severity, 
                                      st.session_state.severity_description, 
                                      st.session_state.length, 
                                      st.session_state.width):
                        st.success(f"Report successfully sent to {email}")
                    else:
                        st.error("Failed to send email. Please try again later.")
                        st.info("If you continue to have problems, please check your email credentials in the .env file.")
            else:
                st.success("No cracks detected in the image.")
                st.markdown("Your property appears to be in good condition based on this image.")

# Add a footer
st.markdown("---")
st.markdown("HomeSafe AI © 2025 | AI-Powered Building Inspection Technology")