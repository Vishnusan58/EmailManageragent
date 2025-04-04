import imaplib
import email
from email.header import decode_header
from email.message import EmailMessage
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import tiktoken
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient
import streamlit as st
import time
import requests
import json

# MCP Server URL
MCP_SERVER_URL = "http://localhost:8000"


# ===== AGENT CLASSES =====
class ClassificationAgent:
    def __init__(self, llm):
        self.chain = (
                PromptTemplate(
                    input_variables=["email_text"],
                    template="""
                Classify this email into EXACTLY ONE category. Be strict. Options:
                Urgent: Immediate action required (security alerts, deadlines <24h)
                Work: Business/professional communications
                Personal: Non-work from known contacts
                Promotions: Commercial content
                Email: {email_text}
                Final Category (ONLY ONE WORD - Urgent/Work/Personal/Promotions):
                """
                ) | llm
        )

    def execute(self, email):
        return self.chain.invoke({"email_text": email['body']}).strip()


class ResponseAgent:
    def __init__(self, llm, openai_client):
        self.llm = llm
        self.openai_client = openai_client
        self.modify_chain = (
                PromptTemplate(
                    input_variables=["original_response", "changes_requested"],
                    template="""
                Modify this response FROM MY PERSPECTIVE:
                Changes: {changes_requested}
                Original: {original_response}
                Modified:
                """
                ) | llm
        )

    def get_mcp_tools(self):
        try:
            response = requests.get(f"{MCP_SERVER_URL}/tools")
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            st.error(f"Failed to fetch MCP tools: {str(e)}")
            return []

    def execute_mcp_tool(self, tool_name, params):
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}/execute",
                json={"tool_name": tool_name, "parameters": params}
            )
            return response.json() if response.status_code == 200 else {"error": "Tool execution failed"}
        except Exception as e:
            st.error(f"Failed to execute MCP tool: {str(e)}")
            return {"error": "Tool execution failed"}

    def generate_response(self, email):
        summary = f"Subject: {email['subject']}\n{email['body'][:500]}"
        tools = self.get_mcp_tools()

        # Initial LLM call with tools
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Draft a concise, polite response to this email:\n{summary}"}],
            tools=tools,
            tool_choice="auto"
        )

        # Handle tool calls if any
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            params = json.loads(tool_call.function.arguments)
            tool_result = self.execute_mcp_tool(tool_name, params)

            # Second LLM call with tool results
            final_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Draft a concise, polite response to this email:\n{summary}"},
                    {"role": "system", "content": f"Tool result: {json.dumps(tool_result)}"}
                ]
            )
            return final_response.choices[0].message.content
        return response.choices[0].message.content

    def modify_response(self, original, changes):
        return self.modify_chain.invoke({
            "original_response": original,
            "changes_requested": changes
        })


class SearchAgent:
    def __init__(self, index):
        self.index = index

    def index_email(self, email, embedding, metadata):
        try:
            self.index.upsert([(str(email['id']), embedding, metadata)])
            return True
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")
            return False

    def query(self, query_embedding, top_k=5):
        try:
            return self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )["matches"]
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []


class SpamFilter:
    def handle(self, mail_conn, email_id):
        try:
            mail_conn.store(email_id, '+X-GM-LABELS', '\\Spam')
            return True
        except Exception as e:
            st.error(f"Failed to mark as spam: {str(e)}")
            return False


# ===== ORCHESTRATOR =====
class EmailOrchestrator:
    def __init__(self, llm, index, mail_conn, openai_api_key):
        self.agents = {
            'classifier': ClassificationAgent(llm),
            'responder': ResponseAgent(llm, OpenAIClient(api_key=openai_api_key)),
            'search': SearchAgent(index),
            'spam_filter': SpamFilter(),
            'mail': mail_conn
        }
        self.embedding_client = OpenAIClient(api_key=openai_api_key)

    def generate_embedding(self, text):
        try:
            response = self.embedding_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            return None

    def process_email(self, email_id, msg):
        try:
            subject = ""
            for part, encoding in decode_header(msg["Subject"]):
                subject += part.decode(encoding or "utf-8") if isinstance(part, bytes) else str(part)

            sender = msg.get("From", "")
            body = self._extract_body(msg)
            clean_text = self._truncate_text(body)

            email_data = {
                'id': email_id,
                'subject': subject,
                'sender': sender,
                'body': clean_text
            }

            result = {'email': email_data}
            category = self.agents['classifier'].execute(email_data)

            if category == "Urgent":
                result.update(self._handle_urgent(email_data))
            elif category == "Work":
                result.update(self._handle_work(email_data))
            elif category == "Promotions":
                result.update(self._handle_promotions(email_id))

            result['category'] = category

            if category != "Promotions":
                embedding = self.generate_embedding(email_data['body'])
                if embedding:
                    metadata = {
                        'category': category,
                        'sender': email_data['sender'],
                        'subject': email_data['subject'],
                        'content': email_data['body'][:1000]
                    }
                    self.agents['search'].index_email(email_data, embedding, metadata)

            return result

        except Exception as e:
            st.error(f"Email processing failed: {str(e)}")
            return {'error': str(e)}

    def _handle_urgent(self, email):
        email['summary'] = f"Subject: {email['subject']}\n{email['body'][:500]}"
        email['drafted_response'] = self.agents['responder'].generate_response(email)
        return {
            'action': 'draft',
            'response': email['drafted_response']
        }

    def _handle_work(self, email):
        return {'action': 'store'}

    def _handle_promotions(self, email_id):
        success = self.agents['spam_filter'].handle(self.agents['mail'], email_id)
        return {
            'action': 'move',
            'success': success
        }

    def _extract_body(self, msg):
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            return part.get_payload(decode=True).decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            return part.get_payload(decode=True).decode('latin-1', errors='replace')
                    elif content_type == "text/html":
                        try:
                            html_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            html_content = part.get_payload(decode=True).decode('latin-1', errors='replace')
                        soup = BeautifulSoup(html_content, "html.parser")
                        return soup.get_text(separator=' ', strip=True)
            else:
                try:
                    return msg.get_payload(decode=True).decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    return msg.get_payload(decode=True).decode('latin-1', errors='replace')
        except Exception as e:
            st.error(f"Error extracting email body: {str(e)}")
            return ""

    def _truncate_text(self, text, max_tokens=3000):
        try:
            if not text or not isinstance(text, str):
                return ""
            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                return encoding.decode(tokens)
            return text
        except Exception as e:
            st.error(f"Error truncating text: {str(e)}")
            return text[:max_tokens * 4]


# ===== STREAMLIT APP =====
def main():
    if 'emails' not in st.session_state:
        st.session_state.emails = {
            "Urgent": [],
            "Work": [],
            "Personal": [],
            "Promotions": []
        }
        st.session_state.modified_responses = {}
        st.session_state.processed = False
        st.session_state.query_results = []

    try:
        openai_api_key = "your-openai-api-key"  # Replace with your key
        llm = OpenAI(api_key=openai_api_key)
        pc = Pinecone(api_key="your-pinecone-api-key")  # Replace with your key
        index = pc.Index("work-mail")
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login("your-email@gmail.com", "your-app-password")  # Replace credentials
        mail.select("inbox")

        orchestrator = EmailOrchestrator(llm, index, mail, openai_api_key)
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return

    st.title("📧 Smart Email Manager with MCP")

    if st.button("🔄 Process Emails"):
        process_emails(orchestrator)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Urgent ⚠️", "Work 💼", "Personal 👤", "Promotions 🎁"
    ])

    with tab1:
        st.subheader("Urgent Emails")
        if st.session_state.emails["Urgent"]:
            for email in st.session_state.emails["Urgent"]:
                with st.expander(f"{email['subject']} - From: {email['sender']}"):
                    st.write(f"**Date:** {email.get('date', 'N/A')}")
                    st.write(f"**Summary:** {email.get('summary', 'No summary available')}")
                    current_response = st.session_state.modified_responses.get(
                        email['id'],
                        email.get('drafted_response', '')
                    )
                    st.text_area("Drafted Response", value=current_response, key=f"response_{email['id']}")
                    changes = st.text_input(
                        "Modifications (if any)",
                        key=f"changes_{email['id']}",
                        help="Describe how you'd like to modify this response"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update Response", key=f"update_{email['id']}"):
                            if changes.strip():
                                modified = orchestrator.agents['responder'].modify_response(
                                    current_response,
                                    changes
                                )
                                st.session_state.modified_responses[email['id']] = modified
                                st.rerun()
                    with col2:
                        if st.button("💾 Save Draft", key=f"save_{email['id']}"):
                            final_response = st.session_state.modified_responses.get(
                                email['id'],
                                email.get('drafted_response', '')
                            )
                            if create_draft(
                                    "your-email@gmail.com",
                                    email['sender'],
                                    f"Re: {email['subject']}",
                                    final_response
                            ):
                                st.success("Draft saved successfully!")
                            else:
                                st.error("Failed to save draft")
        else:
            st.info("No urgent emails found")

    with tab2:
        st.subheader("Work Emails")
        if st.session_state.emails["Work"]:
            for email in st.session_state.emails["Work"]:
                with st.expander(email['subject']):
                    st.write(f"**From:** {email['sender']}")
                    st.write(f"**Date:** {email.get('date', 'N/A')}")
                    st.write(f"**Summary:** {email.get('summary', 'No summary available')}")
            st.subheader("🔍 Search Work Emails")
            query_text = st.text_input("Enter search query:", key="work_query")
            if st.button("Search", key="search_work"):
                if query_text.strip():
                    embedding = orchestrator.generate_embedding(query_text)
                    if embedding:
                        st.session_state.query_results = orchestrator.agents['search'].query(embedding)
            if st.session_state.query_results:
                st.success(f"Found {len(st.session_state.query_results)} similar emails:")
                for match in st.session_state.query_results:
                    with st.expander(f"📄 {match['metadata']['subject']}"):
                        st.write(f"**From:** {match['metadata']['sender']}")
                        st.write(f"**Date:** {match['metadata'].get('date', 'N/A')}")
                        st.write(f"**Content:** {match['metadata']['content'][:500]}...")
        else:
            st.info("No work emails found")

    with tab3:
        st.subheader("Personal Emails")
        if st.session_state.emails["Personal"]:
            for email in st.session_state.emails["Personal"]:
                with st.expander(email['subject']):
                    st.write(f"**From:** {email['sender']}")
                    st.write(f"**Date:** {email.get('date', 'N/A')}")
                    st.write(f"**Summary:** {email.get('summary', 'No summary available')}")
        else:
            st.info("No personal emails found")

    with tab4:
        st.subheader("Promotional Emails")
        if st.session_state.emails["Promotions"]:
            st.info(f"{len(st.session_state.emails['Promotions'])} promotional emails moved to Spam")
            for email in st.session_state.emails["Promotions"]:
                with st.expander(email['subject']):
                    st.write(f"**From:** {email['sender']}")
                    st.write(f"**Date:** {email.get('date', 'N/A')}")
                    st.write(f"**Summary:** {email.get('summary', 'No summary available')}")
        else:
            st.info("No promotional emails found")


def process_emails(orchestrator):
    try:
        status, messages = orchestrator.agents['mail'].search(None, "ALL")
        if status != "OK":
            st.error("Email search failed")
            return
        email_ids = messages[0].split()[-30:]
        category_counts = {cat: 0 for cat in st.session_state.emails.keys()}
        with st.spinner(f"Processing {len(email_ids)} emails..."):
            for email_id in email_ids:
                status, msg_data = orchestrator.agents['mail'].fetch(email_id, "(RFC822)")
                if status == "OK":
                    msg = email.message_from_bytes(msg_data[0][1])
                    result = orchestrator.process_email(email_id, msg)
                    if 'error' not in result:
                        category = result['category']
                        st.session_state.emails[category].append(result['email'])
                        category_counts[category] += 1
                        if category == "Urgent":
                            st.session_state.modified_responses[email_id] = result.get('response', '')
        cols = st.columns(4)
        for idx, category in enumerate(category_counts):
            cols[idx].metric(category, category_counts[category])
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")


def create_draft(sender, to, subject, body):
    try:
        draft_msg = EmailMessage()
        draft_msg["From"] = sender
        draft_msg["To"] = to
        draft_msg["Subject"] = subject
        draft_msg.set_content(body)
        draft_bytes = draft_msg.as_bytes()
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login("your-email@gmail.com", "your-app-password")
        mail.append('[Gmail]/Drafts', '', imaplib.Time2Internaldate(time.time()), draft_bytes)
        return True
    except Exception as e:
        st.error(f"Failed to create draft: {str(e)}")
        return False


if __name__ == "__main__":
    main()