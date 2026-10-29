---
name: coding-review
description: Guidelines for code review, auditing, and debugging.
---

# Coding Principles

The following instructions should always be followed when reviewing code or otherwise performing a comprehensive code audit.

## Instructions

When you are asked to familiarize yourself with a codebase for a code review or code audit, what you should do is create a markdown file and copy the scripts, src, jobs, notebooks, etc. into the markdown file, and remove ALL of the docstrings and comments. Then you should try to understand what all the code does. The reason for removing comments is so that you can get an objective look at the code without being primed by what the comments say the code SHOULD do. You should concisely but clearly summarize the intentions and functionality of the codebase, and you should diagnose anywhere that makes it seem like there might be a bug, functionality error, or similar funky and suspicious issues. Then get rid of the MD file you made (we don't need to clog up our directories with these) and check your assessment against the comments and docstrings actually in the code to find any discrepancies between your understanding of the functionality and the intended functionality of the code.