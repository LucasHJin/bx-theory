"""Callbacks for handling file uploads in ADK web.

These callbacks intercept uploaded files before they're sent to the model,
save them locally, and prevent the raw file content from overwhelming the context.
"""

import os
import tempfile
import base64

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


# Session-specific directory for uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "study_planner_uploads")


def _get_session_upload_dir(session_id: str) -> str:
    """Get or create the upload directory for a session."""
    upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def _save_file_part(upload_dir: str, part: types.Part, index: int) -> tuple[str | None, str | None]:
    """Save a file part to disk and return (saved_path, original_filename)."""
    # Check if this part contains inline data (uploaded file)
    if hasattr(part, 'inline_data') and part.inline_data:
        data = part.inline_data
        mime_type = getattr(data, 'mime_type', 'application/octet-stream')
        content = getattr(data, 'data', None)

        if content and 'pdf' in mime_type.lower():
            filename = f"uploaded_{index}.pdf"
            filepath = os.path.join(upload_dir, filename)

            # Decode if base64 string
            if isinstance(content, str):
                content = base64.b64decode(content)

            with open(filepath, 'wb') as f:
                f.write(content)

            print(f"[Callback] Saved uploaded file: {filepath} ({len(content)} bytes)")
            return filepath, filename

    # Check for file_data (Gemini File API reference)
    if hasattr(part, 'file_data') and part.file_data:
        uri = getattr(part.file_data, 'file_uri', None)
        if uri:
            print(f"[Callback] Found file reference: {uri}")
            return uri, uri

    return None, None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Intercept LLM request before it goes to the model.

    This callback runs on EVERY request and:
    1. Saves any new file parts to disk (first time only)
    2. Strips ONLY file content (inline_data, file_data) to prevent token overflow
    3. Keeps all other parts: text, function_call, function_response

    Args:
        callback_context: Context with state, session, etc.
        llm_request: The request being sent to the model

    Returns:
        None to continue with request, or LlmResponse to skip model call
    """
    if not llm_request or not llm_request.contents:
        return None

    # Get session ID for upload directory
    session_id = "default"
    if callback_context.session:
        session_id = str(getattr(callback_context.session, 'id', 'default'))

    upload_dir = _get_session_upload_dir(session_id)

    # Only save files on first pass
    first_pass = not callback_context.state.get("_files_processed")
    saved_files: list[str] = []

    # Strip ONLY file content (inline_data, file_data) from requests
    # Keep everything else: text, function_call, function_response, etc.
    new_contents: list[types.Content] = []

    for content in llm_request.contents:
        if not hasattr(content, 'parts') or not content.parts:
            new_contents.append(content)
            continue

        new_parts: list[types.Part] = []
        for i, part in enumerate(content.parts):
            # Check if this is a file part (the only thing we want to strip)
            has_inline_data = hasattr(part, 'inline_data') and part.inline_data
            has_file_data = hasattr(part, 'file_data') and part.file_data

            if has_inline_data:
                # Save file on first pass, then skip this part
                if first_pass:
                    saved_path, _ = _save_file_part(upload_dir, part, len(saved_files))
                    if saved_path:
                        saved_files.append(saved_path)
                # Don't add file parts to new_parts
                continue

            if has_file_data:
                # Skip file_data parts (references to uploaded files)
                continue

            # Keep ALL other parts (text, function_call, function_response, etc.)
            new_parts.append(part)

        # Rebuild content with file parts removed
        if new_parts:
            role = getattr(content, 'role', 'user')
            new_contents.append(types.Content(role=role, parts=new_parts))

    # Update state on first pass
    if first_pass and saved_files:
        callback_context.state["uploaded_files"] = saved_files
        callback_context.state["session_id"] = session_id
        callback_context.state["_files_processed"] = True
        print(f"[Callback] Saved {len(saved_files)} files to {upload_dir}")

    # Replace contents with stripped version
    if new_contents:
        llm_request.contents = new_contents

    print(f"[Callback] Processed request: {len(new_contents)} content(s), files stripped")

    # Continue with the (modified) request
    return None
