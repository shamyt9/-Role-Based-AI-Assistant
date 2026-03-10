import streamlit as st
from langchain_core.prompts import ChatPromptTemplate 
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import re

# ========== CONFIGURATION ==========
os.environ['HF_HOME'] = 'E:/LLM (LANGUAGE MODELS)/HuggingFace_cache'

# ========== RESPONSE CLEANING FUNCTIONS ==========
def clean_tinyllama_response(response_text):
    """
    Clean TinyLlama response by removing special tokens and unwanted patterns
    """
    # Remove user prompts
    cleaned = re.sub(r'<\|user\|>.*?</s>\s*', '', response_text, flags=re.DOTALL)
    
    # Remove assistant tokens
    cleaned = re.sub(r'<\|assistant\|>\s*', '', cleaned)
    
    # Remove end tokens
    cleaned = re.sub(r'</s>\s*$', '', cleaned)
    
    # Remove references/sources sections
    cleaned = re.sub(r'\n\s*References?:.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n\s*Sources?:.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove specific patterns (customize as needed)
    cleaned = re.sub(r'\n\s*Capital of India can also be found in:.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Final strip
    return cleaned.strip()

def get_clean_response(model, prompt_text):
    """
    Get response from model and clean it in one function
    """
    res = model.invoke(prompt_text)
    return clean_tinyllama_response(res.content)

# ========== MODEL LOADING FUNCTION ==========
@st.cache_resource
def load_model():
    """
    Load TinyLlama model with caching to avoid reloading
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs=dict(
            temperature=0.1,
            max_new_tokens=200
        )
    )
    return ChatHuggingFace(llm=llm)

# ========== PROMPT TEMPLATE FUNCTIONS ==========
def get_summarizer_prompt():
    """Return summarizer prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a professional summarizer. Create {length} summaries."),
        ("human", "Summarize this: {text}")
    ])

def get_story_writer_prompt():
    """Return story writer prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a creative story writer. Write {genre} stories for {age_group} readers."),
        ("human", "Write a story about: {topic}")
    ])

def get_translator_prompt():
    """Return translator prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate from {source_lang} to {target_lang}."),
        ("human", "Translate: {text}")
    ])

def get_code_reviewer_prompt():
    """Return code reviewer prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer. Review code for {language} projects."),
        ("human", "Review this code:\n{code}")
    ])

def get_teacher_prompt():
    """Return teacher prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a {subject} teacher for {grade_level} students. Explain in simple terms."),
        ("human", "Explain: {concept}")
    ])

def get_blog_chain_prompts():
    """Return all prompts for blog generation chain"""
    return {
        "outline": ChatPromptTemplate.from_messages([
            ("system", "You are a professional content strategist. Create detailed blog outlines."),
            ("human", "Create a detailed outline for a blog post about '{topic}' for {audience}. Include introduction, 3-4 main points, and conclusion.")
        ]),
        "draft": ChatPromptTemplate.from_messages([
            ("system", "You are a blog writer. Write in a {tone} tone."),
            ("human", "Using this outline, write a complete blog post about '{topic}':\n\n{outline}")
        ]),
        "polish": ChatPromptTemplate.from_messages([
            ("system", "You are an expert editor. Polish and refine blog posts for clarity and engagement."),
            ("human", "Polish this blog post about '{topic}' for {audience} with {tone} tone. Fix any issues and make it flow better:\n\n{draft}")
        ])
    }

# ========== PROCESSING FUNCTIONS ==========
def process_summarizer(model, text, length):
    """Process summarizer request"""
    template = get_summarizer_prompt()
    messages = template.format_messages(length=length, text=text)
    return get_clean_response(model, messages)

def process_story_writer(model, genre, age_group, topic):
    """Process story writer request"""
    template = get_story_writer_prompt()
    messages = template.format_messages(genre=genre, age_group=age_group, topic=topic)
    return get_clean_response(model, messages)

def process_translator(model, source_lang, target_lang, text):
    """Process translator request"""
    template = get_translator_prompt()
    messages = template.format_messages(source_lang=source_lang, target_lang=target_lang, text=text)
    return get_clean_response(model, messages)

def process_code_reviewer(model, language, code):
    """Process code reviewer request"""
    template = get_code_reviewer_prompt()
    messages = template.format_messages(language=language, code=code)
    return get_clean_response(model, messages)

def process_teacher(model, subject, grade_level, concept):
    """Process teacher request"""
    template = get_teacher_prompt()
    messages = template.format_messages(subject=subject, grade_level=grade_level, concept=concept)
    return get_clean_response(model, messages)

def process_blog_chain(model, topic, tone, audience):
    """Process blog generation chain with multiple steps"""
    prompts = get_blog_chain_prompts()
    
    # Step 1: Generate outline
    outline_messages = prompts["outline"].format_messages(topic=topic, audience=audience)
    outline = get_clean_response(model, outline_messages)
    
    # Step 2: Write draft from outline
    draft_messages = prompts["draft"].format_messages(topic=topic, tone=tone, outline=outline)
    draft = get_clean_response(model, draft_messages)
    
    # Step 3: Polish draft
    polish_messages = prompts["polish"].format_messages(topic=topic, audience=audience, tone=tone, draft=draft)
    final = get_clean_response(model, polish_messages)
    
    return {
        "outline": outline,
        "draft": draft,
        "final": final
    }

# ========== UI RENDERING FUNCTIONS ==========
def render_header():
    """Render app header"""
    st.title("🎭 Role-Based AI Assistant")
    st.markdown("---")


def render_summarizer_ui(model):
    """Render summarizer UI"""
    text = st.text_area("Enter text to summarize:", height=150)
    length = st.selectbox("Summary length:", ["short", "medium", "detailed"])
    
    if st.button("Summarize"):
        if text:
            with st.spinner("Summarizing..."):
                response = process_summarizer(model, text, length)
                st.success("### Summary:")
                st.write(response)
        else:
            st.warning("Please enter some text!")

def render_story_writer_ui(model):
    """Render story writer UI"""
    genre = st.selectbox("Genre:", ["Fantasy", "Sci-Fi", "Mystery", "Comedy", "Adventure"])
    age_group = st.selectbox("Age group:", ["Children (5-8)", "Middle Grade (9-12)", "Teen", "Adult"])
    topic = st.text_input("Story topic/idea:")
    
    if st.button("Write Story"):
        if topic:
            with st.spinner("Writing story..."):
                response = process_story_writer(model, genre, age_group, topic)
                st.success("### Story:")
                st.write(response)
        else:
            st.warning("Please enter a topic!")

def render_translator_ui(model):
    """Render translator UI"""
    source_lang = st.text_input("Source language:", "English")
    target_lang = st.text_input("Target language:", "Spanish")
    text = st.text_area("Text to translate:", height=100)
    
    if st.button("Translate"):
        if text:
            with st.spinner("Translating..."):
                response = process_translator(model, source_lang, target_lang, text)
                st.success(f"### Translation ({target_lang}):")
                st.write(response)
        else:
            st.warning("Please enter text to translate!")

def render_code_reviewer_ui(model):
    """Render code reviewer UI"""
    language = st.selectbox("Programming language:", ["Python", "JavaScript", "Java", "C++", "HTML/CSS"])
    code = st.text_area("Paste your code:", height=200)
    
    if st.button("Review Code"):
        if code:
            with st.spinner("Reviewing code..."):
                response = process_code_reviewer(model, language, code)
                st.success("### Code Review:")
                st.write(response)
        else:
            st.warning("Please paste some code!")

def render_teacher_ui(model):
    """Render teacher UI"""
    subject = st.selectbox("Subject:", ["Math", "Science", "History", "Literature", "Art"])
    grade_level = st.selectbox("Grade level:", ["Elementary", "Middle School", "High School", "College"])
    concept = st.text_input("Concept to explain:")
    
    if st.button("Explain"):
        if concept:
            with st.spinner("Explaining..."):
                response = process_teacher(model, subject, grade_level, concept)
                st.success("### Explanation:")
                st.write(response)
        else:
            st.warning("Please enter a concept!")

def render_blog_chain_ui(model):
    """Render blog chain UI"""
    st.subheader("📝 Blog Post Generator (Prompt Chain)")
    
    topic = st.text_input("Blog topic:")
    tone = st.selectbox("Tone:", ["Professional", "Casual", "Humorous", "Inspirational"])
    audience = st.text_input("Target audience:", "General readers")
    
    if st.button("Generate Blog Post", type="primary"):
        if topic:
            with st.spinner("Running 3-step prompt chain..."):
                results = process_blog_chain(model, topic, tone.lower(), audience)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📋 Outline", "📄 Draft", "✨ Final"])
                
                with tab1:
                    st.markdown("### Generated Outline")
                    st.info(results["outline"])
                
                with tab2:
                    st.markdown("### First Draft")
                    st.info(results["draft"])
                
                with tab3:
                    st.markdown("### Polished Final")
                    st.success(results["final"])
                
                st.markdown("---")
                st.caption("✅ Prompt chain complete: Outline → Draft → Polish")
        else:
            st.warning("Please enter a blog topic!")

# ========== MAIN APP ==========
def main():
    """Main application entry point"""
    # Render UI elements
    render_header()
  
    
    # Load model (cached)
    model = load_model()
    
    # Role selector
    role = st.selectbox(
        "Select AI Role:",
        ["Summarizer", "Story Writer", "Translator", "Code Reviewer", "Teacher", "Blog Post Generator (Chain)"]
    )
    
    # Render appropriate UI based on role
    if role == "Summarizer":
        render_summarizer_ui(model)
    elif role == "Story Writer":
        render_story_writer_ui(model)
    elif role == "Translator":
        render_translator_ui(model)
    elif role == "Code Reviewer":
        render_code_reviewer_ui(model)
    elif role == "Teacher":
        render_teacher_ui(model)
    elif role == "Blog Post Generator (Chain)":
        render_blog_chain_ui(model)
    
    # Footer
    st.markdown("---")
   

if __name__ == "__main__":
    main()