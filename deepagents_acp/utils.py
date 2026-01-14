from acp.schema import (
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)


def convert_text_block_to_content_blocks(block: TextContentBlock):
    return [{"type": "text", "text": block.text}]


def convert_image_block_to_content_blocks(block: ImageContentBlock, *, root_dir: str):
    # Image blocks contain visual data
    # URIs are local file paths, provide as text so agent can read with tools
    if block.uri:
        # Truncate root_dir from path while preserving file:// prefix
        uri = block.uri
        has_file_prefix = uri.startswith("file://")
        if has_file_prefix:
            path = uri[7:]  # Remove "file://" temporarily
        else:
            path = uri

        # Remove root_dir prefix to get path relative to agent's working directory
        if path.startswith(root_dir):
            path = path[len(root_dir) :].lstrip("/")

        # Restore file:// prefix if it was present
        uri = f"file://{path}" if has_file_prefix else path
        return [
            {
                "type": "text",
                "text": f"[Image file at path: {uri}, MIME type: {block.mime_type}]",
            }
        ]
    else:
        # Use inline base64 data
        data_uri = f"data:{block.mime_type};base64,{block.data}"
        return [{"type": "image_url", "image_url": data_uri}]


def convert_audio_block_to_content_blocks(block: AudioContentBlock):
    raise Exception("Audio is not currently supported.")


def convert_resource_block_to_content_blocks(
    block: ResourceContentBlock, *, root_dir: str
):
    # Resource blocks reference external resources
    resource_text = f"[Resource: {block.name}"
    if block.uri:
        # Truncate root_dir from path while preserving file:// prefix
        uri = block.uri
        has_file_prefix = uri.startswith("file://")
        if has_file_prefix:
            path = uri[7:]  # Remove "file://" temporarily
        else:
            path = uri

        # Remove root_dir prefix to get path relative to agent's working directory
        if path.startswith(root_dir):
            path = path[len(root_dir) :].lstrip("/")

        # Restore file:// prefix if it was present
        uri = f"file://{path}" if has_file_prefix else path
        resource_text += f"\nURI: {uri}"
    if block.description:
        resource_text += f"\nDescription: {block.description}"
    if block.mime_type:
        resource_text += f"\nMIME type: {block.mime_type}"
    resource_text += "]"
    return [{"type": "text", "text": resource_text}]


def convert_embedded_resource_block_to_content_blocks(
    block: EmbeddedResourceContentBlock,
) -> list[dict]:
    # Embedded resource blocks contain the resource data inline
    resource = block.resource
    if hasattr(resource, "text"):
        mime_type = getattr(resource, "mime_type", "application/text")
        return [
            {"type": "text", "text": f"[Embedded {mime_type} resource: {resource.text}"}
        ]
    elif hasattr(resource, "blob"):
        mime_type = getattr(resource, "mime_type", "application/octet-stream")
        data_uri = f"data:{mime_type};base64,{resource.blob}"
        return [
            {
                "type": "text",
                "text": f"[Embedded resource: {data_uri}]",
            }
        ]
    else:
        raise Exception(
            "Could not parse embedded resource block. Block expected either a `text` or `blob` property."
        )
