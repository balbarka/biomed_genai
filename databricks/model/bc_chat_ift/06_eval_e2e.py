# Databricks notebook source
This notebook is sixth in a series that generates synthetic data for subsequent chat completion Fine Tuning (FT). This notebook compares the the performance of the whole RAG system with either the finetuned model or the base model as the answering LLM.

What this notebook does:
1. Create RAG system with base model as the answering LLM.
2. Create RAG system with finetuned model as the answering LLM.
3. Compare the performance of the two RAG systems using mlflow.evaluate
