import gradio as gr

def run_triage(letter):
    # This function is a placeholder for your triage classification process
    return f"Hip"

def send_to_clinic(clinic_type, clinician):
    return f"Sending {clinic_type} case to {clinician}"

# Options for clinic type
clinic_options = ["Primary Arthroplasty", "Revision Arthroplasty", "Soft Tissue", "Further Info Required"]

# Options for clinicians
clinician_options = ["Mr. Osman", "Prof. Sochart", "Mr. Asopa", "Mr. Radha"]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Triage Agent")
    
    # Referral Letter
    referral_input = gr.File(label="Drop the Referral Letter")
    
    # Triage button
    triage_button = gr.Button("Run Triage", variant="primary")
    triage_output = gr.Textbox(label="Clinic")
    
    triage_button.click(fn=run_triage, inputs=referral_input, outputs=triage_output)
    
    # Clinic type selection
    clinic_type = gr.Radio(clinic_options, label="Select Clinic Type", interactive=True)
    
    # Clinician selection
    clinician = gr.Radio(clinician_options, label="Select Clinician", interactive=True)
    
    # Send to clinic button
    send_button = gr.Button("Send to Clinic", variant="primary")
    send_output = gr.Textbox(label="Send Output")
    
    send_button.click(fn=send_to_clinic, inputs=[clinic_type, clinician], outputs=send_output)

demo.launch()
