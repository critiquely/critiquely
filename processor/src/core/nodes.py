import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse
from uuid import uuid4

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError
from github import Auth, Github
from github.GithubException import GithubException
from langchain_core.messages import HumanMessage
from langgraph.graph import END

from src.config import settings
from src.core.state import DevAgentState
from src.utils.fs import get_temp_dir
from src.utils.git import create_github_https_url
from src.utils.state import get_state_value

# Use the root logger configuration from CLI
logger = logging.getLogger(__name__)


###############################################################################
#                          SHARED UTILITIES AND BASE CLASSES                 #
###############################################################################


def strip_markdown_json(content: str) -> str:
    """
    Strip markdown code block formatting from JSON responses.

    Handles cases where the LLM wraps JSON in markdown code blocks like:
    ```json
    {...}
    ```

    Args:
        content: The raw response content that may contain markdown formatting

    Returns:
        The cleaned JSON string
    """
    content = content.strip()

    # Check if content starts with markdown code block
    if content.startswith("```"):
        # Find the end of the first line (which contains ```json or just ```)
        first_newline = content.find("\n")
        if first_newline != -1:
            # Remove the opening ```json or ``` line
            content = content[first_newline + 1:]

        # Remove the closing ``` line
        if content.endswith("```"):
            content = content[:-3].rstrip()

    return content.strip()



class GitOperationError(Exception):
    """Custom exception for Git operations"""
    pass


@dataclass
class NodeResult:
    """Standardized return type for all nodes"""
    success: bool
    message: str
    updates: Optional[dict] = None
    error: Optional[str] = None
    
    def to_state_update(self) -> dict:
        """Convert to state update dictionary"""
        result = {"messages": [HumanMessage(content=self.message)]}
        if self.updates:
            result.update(self.updates)
        return result


class GitOperations:
    """Shared Git operations with consistent error handling"""
    
    @staticmethod
    def get_repo(clone_path: str) -> Repo:
        """Open a repo with standardized error handling"""
        try:
            return Repo(clone_path)
        except (NoSuchPathError, InvalidGitRepositoryError) as e:
            msg = f"❌ Error: Cannot open repo at '{clone_path}': {e}"
            logger.error(msg)
            raise GitOperationError(msg)
    
    @staticmethod
    def setup_remote_with_token(repo: Repo, repo_url: str) -> Any:
        """Setup remote URL with token injection"""
        git_url = create_github_https_url(repo_url)
        origin = repo.remote(name="origin")
        origin.set_url(git_url)
        return origin
    
    @staticmethod
    def get_state_values(state: DevAgentState, *keys: str) -> tuple:
        """Get multiple state values at once"""
        return tuple(get_state_value(state, key) for key in keys)


class BaseNode:
    """Base class for all nodes with common functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_success_result(self, message: str, updates: Optional[dict] = None) -> NodeResult:
        """Create a successful node result"""
        self.logger.info(message)
        return NodeResult(success=True, message=message, updates=updates)
    
    def create_error_result(self, error: str) -> NodeResult:
        """Create an error node result"""
        self.logger.error(error)
        return NodeResult(success=False, message=error, error=error)
    
    def handle_git_error(self, operation: str, exc: Exception) -> NodeResult:
        """Standardized Git error handling"""
        error = f"❌ Failed to {operation}: {exc}"
        return self.create_error_result(error)

###############################################################################
#                                G I T   N O D E S                            #
###############################################################################


# --- Node: Clone Repo ---
def clone_repo(state: DevAgentState) -> dict:
    repo_url = get_state_value(state, "repo_url")
    branch = get_state_value(state, "base_branch")

    git_url = create_github_https_url(repo_url)

    try:
        temp_dir = tempfile.mkdtemp(dir=get_temp_dir())
        logger.info(f"🔄 Cloning {repo_url}@{branch} into {temp_dir}")
        repo = Repo.clone_from(
            git_url, temp_dir, branch=branch, depth=1, single_branch=True
        )
        # Scrub token from remote config
        repo.remote().set_url(repo_url)

        msg = f"✅ Cloned {repo_url}@{branch} to {temp_dir}"
        logger.info(msg)
        return {"clone_path": temp_dir, "messages": [HumanMessage(content=msg)]}

    except GitCommandError as exc:
        error = f"❌ Failed to clone {repo_url}@{branch}: {exc}"
        logger.error(error)
        return {"messages": [HumanMessage(content=error)]}


# --- Node: Create Branch ---
def create_branch(state: DevAgentState) -> DevAgentState:
    clone_path = get_state_value(state, "clone_path")
    branch = get_state_value(state, "base_branch")

    try:
        repo = Repo(clone_path)
    except (NoSuchPathError, InvalidGitRepositoryError) as e:
        msg = f"❌ Error: Cannot open repo at '{clone_path}': {e}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    branch_name = f"critiquely/{branch}-improvements-{uuid4().hex[:8]}"
    try:
        logger.info(f"🔄 Creating a new branch: {branch_name}")
        branch = repo.create_head(branch_name)
        branch.checkout()

        msg = f"✅ New branch created: {branch_name}"
        logger.info(msg)
        return {
            "new_branch": branch_name,
            "messages": [HumanMessage(content=msg)],
        }

    except GitCommandError as exc:
        error = f"❌ Failed to create {branch_name}: {exc}"
        logger.error(error)
        return {"messages": [HumanMessage(content=error)]}


# --- Node: Create Branch ---
def commit_code(state: DevAgentState) -> DevAgentState:
    clone_path = get_state_value(state, "clone_path")
    branch = get_state_value(state, "new_branch")
    repo_url = get_state_value(state, "repo_url")
    current_recommendation = get_state_value(state, "current_recommendation")

    git_url = create_github_https_url(repo_url)
    recommendation_summary = current_recommendation.get("summary", [])

    # 2) Open the repo
    try:
        repo = Repo(clone_path)
    except (NoSuchPathError, InvalidGitRepositoryError) as e:
        msg = f"❌ Error: Cannot open repo at '{clone_path}': {e}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    # 3) Inject token into the origin URL
    origin = repo.remote(name="origin")
    origin.set_url(git_url)

    # Stage & commit
    try:
        # Stage all changes (new, modified, deleted)
        repo.git.add("--all")
        # Create a commit
        repo.index.commit(recommendation_summary)
        msg = f"📝 Committed changes to '{branch}'"
        logger.info(msg)
        return {"messages": [HumanMessage(content=msg)]}
    except GitCommandError as exc:
        msg = f"❌ Failed to add/commit changes: {exc}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}


# --- Node: Push Code ---
def push_code(state: DevAgentState) -> DevAgentState:
    clone_path = get_state_value(state, "clone_path")
    branch = get_state_value(state, "new_branch")
    repo_url = get_state_value(state, "repo_url")

    git_url = create_github_https_url(repo_url)

    # 2) Open the repo
    try:
        repo = Repo(clone_path)
    except (NoSuchPathError, InvalidGitRepositoryError) as e:
        msg = f"❌ Error: Cannot open repo at '{clone_path}': {e}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    # 3) Inject token into the origin URL
    origin = repo.remote(name="origin")
    origin.set_url(git_url)

    # 4) Push
    try:
        logger.info(f"🔄 Pushing branch '{branch}' to origin")
        origin.push(refspec=f"{branch}:{branch}")
        msg = f"✅ Pushed branch '{branch}' to origin"
        logger.info(msg)
        return {"new_branch": branch, "messages": [HumanMessage(content=msg)]}

    except GitCommandError as exc:
        error = f"❌ Failed to push to '{branch}': {exc}"
        logger.error(error)
        return {"messages": [HumanMessage(content=error)]}

# --- Node: Clone Repo ---
def pr_repo(state: DevAgentState) -> dict:
    repo_url = get_state_value(state, "repo_url")
    base_branch = get_state_value(state, "base_branch")
    head_branch = get_state_value(state, "new_branch")

    title = f"Critiquely improvements"
    body = "Automated code review fixes and improvements."

    # Retrieve GitHub Token
    if not settings.github_token:
        msg = "❌ GITHUB_TOKEN is unset or empty."
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    # Retrieve repo name from URL
    repo_name = urlparse(repo_url).path.lstrip("/").removesuffix(".git")

    # Access the GitHub Repo
    try:
        gh = Github(auth=Auth.Token(settings.github_token))
        repo = gh.get_repo(repo_name)
    except GithubException as exc:
        msg = f"❌ Failed to access repo '{repo_name}': {exc.data.get('message', exc)}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    try:
        pr = repo.create_pull(
            base=base_branch,
            head=head_branch,
            title=title,
            body=body,
            draft=False,
        )
        msg = f"✅ Opened PR #{pr.number}: {pr.html_url}"
        logger.info(msg)
        return {
            "pr_number": pr.number,
            "pr_url": pr.html_url,
            "messages": [HumanMessage(content=msg)],
        }

    except GithubException as exc:
        msg = (
            f"❌ Failed to open PR {head_branch} → {base_branch}: "
            f"{exc.data.get('message', exc)}"
        )
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}


def comment_on_original_pr(state: DevAgentState) -> dict:
    """
    comment_on_original_pr(state: DevAgentState) -> dict
    This node creates a comment on the GitHub pull request that initiated the review process.
    """
    original_pr_url = state.get("original_pr_url")

    # Retrieve GitHub Token
    if not settings.github_token:
        msg = "❌ GITHUB_TOKEN is unset or empty."
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    # Parse original PR URL to extract repo and PR number
    try:
        # Expected format: https://github.com/owner/repo/pull/123
        url_parts = original_pr_url.rstrip("/").split("/")
        if len(url_parts) < 7 or url_parts[-2] != "pull":
            raise ValueError("Invalid PR URL format")

        repo_name = f"{url_parts[-4]}/{url_parts[-3]}"
        pr_number = int(url_parts[-1])
    except (ValueError, IndexError) as e:
        msg = f"❌ Failed to parse PR URL '{original_pr_url}': {e}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    try:
        gh = Github(auth=Auth.Token(settings.github_token))
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
    except Exception as exc:
        msg = f"❌ Failed to access PR #{pr_number} in '{repo_name}': {exc}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}

    # TODO: Add a summary of the changes made by the Critiquely review
    comment_body = (
        f"🤖 **Critiquely Review Complete**\n\n"
        f"**Review PR:** {state.get("pr_url")}\n\n"
        f"The improvements include automated code review fixes and enhancements. "
        f"Please review the changes and merge if they look good!"
    )

    try:
        pr.create_issue_comment(comment_body)
        msg = f"✅ Commented on original PR {original_pr_url}"
        logger.info(msg)
        return {
            "original_pr_url": original_pr_url,
            "messages": [HumanMessage(content=msg)],
        }

    except Exception as exc:
        msg = f"❌ Failed to comment on PR {original_pr_url}: {exc}"
        logger.error(msg)
        return {"messages": [HumanMessage(content=msg)]}


###############################################################################
#                                L L M   N O D E S                            #
###############################################################################


# --- Node: Inspect Files ---
def inspect_files(llm, state: DevAgentState) -> dict:
    if not state.get("modified_files"):
        logger.info("✅ No modified files to inspect. Skipping.")
        return state

    entry = state["modified_files"].pop(0)
    fname = entry.get("filename")
    fpath = Path(state.get("clone_path", "")) / fname

    try:
        file_text = fpath.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"❌ Could not read file {fpath}: {e}")
        state.setdefault("messages", []).append(HumanMessage(content=msg))
        return state

    lines_changed = entry.get("lines_changed", [])
    logger.info(f"🔍 Inspecting {fname} (lines {lines_changed})")

    state.update(
        {
            "active_file_name": fname,
            "active_file_content": file_text,
            "active_file_lines_changed": lines_changed,
        }
    )

    prompt = HumanMessage(
        content=(
            "You are a senior Python reviewer.\n\n"
            f"File: {fpath}\n"
            f"Modified lines: {lines_changed}\n\n"
            "Review only sections directly impacted by the modifications. "
            "Provide up to 3 high-impact recommendations in JSON array only.\n\n"
            "Output format:\n"
            "[{'file':'<filename>','summary':'<github commit style summary using conventional commit syntax>','recommendation':'<recommendation>'},{'file':'<filename>','summary':'<github commit style summary>','recommendation':'<recommendation>'}]\n\n"
            "You can have multiple objects for a specific file if there are multple recommendations\n\n"
            "**DO NOT** include anything other than the JSON list"
            f"File contents:\n{file_text}"
        )
    )
    state.setdefault("messages", []).append(prompt)
    logger.info("💬 Sending review prompt to LLM")
    response = llm.invoke([prompt])
    logger.info("✅ LLM response received")
    state["messages"].append(response)

    try:
        cleaned_content = strip_markdown_json(response.content)
        parsed = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        msg = f"❌ Failed to parse JSON: {e}"
        logger.error(msg)
        parsed = []

    state.setdefault("recommendations", []).extend(parsed)
    return state


def apply_recommendations_with_mcp(
    llm_with_tools, state: DevAgentState
) -> DevAgentState:
    recs_list = state.get("recommendations", [])
    if not recs_list:
        logger.info("✅ No recommendations to apply. Skipping.")
        return state

    current = recs_list.pop(0)
    state.update({"current_recommendation": current})
    file_path = Path(current.get("file", ""))
    recs = current.get("recommendation")

    if not recs:
        logger.warning(f"❌ No recommendations for {file_path}")
        return state
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return state

    file_text = file_path.read_text(encoding="utf-8")
    logger.info(f"🔍 Applying recommendation to {file_path.name}")

    instructions = "\n".join(f"- {r}" for r in recs)
    prompt = HumanMessage(
        content=(
            "You are a coding assistant. A user requested edits to a source file.\n\n"
            f"File path: {file_path}\n\n"
            f"File contents:\n{file_text}\n\n"
            f"Requested changes:\n{instructions}\n\n"
            "Choose and invoke a tool from your toolkit. Return only the invocation."
        )
    )
    state.setdefault("messages", []).append(prompt)
    logger.info("💬 Invoking LLM with tools")

    result = llm_with_tools.invoke([prompt])
    logger.info("✅ Received tool invocation from LLM")
    state["messages"].append(result)

    state.setdefault("updated_files", []).append(str(file_path))
    logger.info(
        f"✅ Applied recommendations to {file_path.name}; {len(recs_list)} remaining"
    )
    return state
