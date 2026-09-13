"""Extraction-recall harness for Context Guard's known-value extractor.

Measures how many planted facts the extractor registers from user and tool
text, with which anchor, and whether a wrong claim about each fact would fire
known-value drift. Ground truth is the planted fact sheet; no model labels
anything. See ``python -m scripts.extraction_recall --help``.
"""
