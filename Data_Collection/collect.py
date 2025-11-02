from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import os
import re

# Load environment variables
load_dotenv()

# ----------------------------
# 1️⃣ Load URLs from Excel file
# ----------------------------
excel_path = "../data/data_links.xlsx"  # 👈 update path if needed
column_name = "URL"                      # 👈 the column in your Excel that has the links

# Read URLs from Excel
df = pd.read_excel(excel_path)
if column_name not in df.columns:
    raise ValueError(f"Column '{column_name}' not found in {excel_path}")

urls = df[column_name].dropna().tolist()  # remove empty cells, convert to list
print(f"✅ Loaded {len(urls)} URLs from Excel")

# ----------------------------
# 2️⃣ Define Model and Prompt
# ----------------------------
model = GoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

prompt = PromptTemplate(
    template='''
You are a precise information extraction assistant. From the following text, extract only the relevant information related to the specified topics and present it in detailed, factual, and well-structured paragraphs.

Instructions:
1. Focus strictly on the content present in the text — do not add assumptions.
2. If information about a topic is missing, state "Not mentioned."
3. Write clear, coherent paragraphs for each topic.
4. Preserve important contextual details (names, relationships, events, etc.) where relevant.

Topics: Name, One-line Introduction, Personality, Powers and Abilities, Weaknesses, History, Appearances

Text:
{text}

Output format:
Name: <name>
One-line Introduction: <introduction>
Appearance: <detailed paragraph>
History: <detailed paragraph>
Weaknesses: <detailed paragraph>
Powers and Abilities: <detailed paragraph>
Personality: <detailed paragraph>
''',
    input_variables=['text']
)

# ----------------------------
# 3️⃣ Create the processing chain
# ----------------------------
chain = prompt | model | parser

# ----------------------------
# 4️⃣ Folder for PDFs
# ----------------------------
pdf_folder = "../data/pdf"
os.makedirs(pdf_folder, exist_ok=True)

# ----------------------------
# 5️⃣ Loop through URLs and generate PDFs
# ----------------------------
for url in urls:
    print(f"\n🔗 Processing: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text = docs[0].page_content

        # Run the extraction chain
        output = chain.invoke({'text': text})

        # Extract alien name
        match = re.search(r"Name:\s*(.*)", output)
        alien_name = match.group(1).strip() if match else url.split("/")[-1].replace("_", " ")

        # Clean name for filename
        file_name = re.sub(r'[<>:"/\\|?*]', '_', alien_name)
        pdf_path = os.path.join(pdf_folder, f"{file_name}.pdf")

        # Create PDF
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        for line in output.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), styles["Normal"]))
                story.append(Spacer(1, 10))

        doc.build(story)

        print(f"✅ PDF saved: {pdf_path}")

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")

print("\n🎉 All PDFs generated successfully!")
