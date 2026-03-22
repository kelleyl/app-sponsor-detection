"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API

    :return: AppMetadata object holding all necessary information.
    """

    metadata = AppMetadata(
        name="Sponsor Detection",
        description="Detects sponsor mentions in video programs by analyzing ASR transcript. "
                    "Extracts transcript from upstream Whisper/Parakeet output, prompts an LLM "
                    "via an OpenAI-compatible API (transformers serve, vllm, ollama, etc.) to "
                    "identify sponsors, then aligns quoted text back to timestamped segments.",
        app_license="Apache 2.0",
        identifier="sponsor-detection",
        url="https://github.com/clamsproject/app-sponsor-detection",
    )

    # Inputs: ASR output (TextDocument with Token+TimeFrame+Alignment for timestamps)
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(DocumentTypes.TextDocument,
                       description="Full transcript from ASR (e.g., Whisper, Parakeet)")
    metadata.add_input(AnnotationTypes.Token, required=False,
                       description="Word-level tokens from ASR")
    metadata.add_input(AnnotationTypes.TimeFrame, required=False,
                       description="Timestamp segments aligned to tokens")
    metadata.add_input(AnnotationTypes.Alignment, required=False,
                       description="Alignments linking tokens to time segments")

    # Outputs
    metadata.add_output(DocumentTypes.TextDocument,
                        description="JSON document with detected sponsor name and time range")
    metadata.add_output(AnnotationTypes.Alignment,
                        description="Links each sponsor detection to the transcript segment it was found in")

    # Parameters
    metadata.add_parameter(
        name='apiUrl',
        description='Base URL of an OpenAI-compatible API server (e.g., transformers serve, vllm, ollama)',
        type='string',
        default='http://localhost:8000'
    )

    metadata.add_parameter(
        name='modelName',
        description='Model name to request from the API server',
        type='string',
        default='Qwen/Qwen3.5-4B'
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
