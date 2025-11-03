"""
Sentiment Analysis Module
FinBERT-based sentiment analysis for financial news
"""
import streamlit as st

# Global variables for model caching
_finbert_model = None
_finbert_tokenizer = None


def load_finbert_model():
    """Load FinBERT model once and cache it"""
    global _finbert_model, _finbert_tokenizer
    
    if _finbert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "ProsusAI/finbert"
            _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _finbert_model.eval()
            
            return _finbert_tokenizer, _finbert_model
        except Exception as e:
            st.warning(f"FinBERT not available: {str(e)[:100]}")
            return None, None
    
    return _finbert_tokenizer, _finbert_model


def get_finbert_sentiment(text: str):
    """Get FinBERT sentiment for text"""
    try:
        import torch
        
        tokenizer, model = load_finbert_model()
        if tokenizer is None or model is None:
            return "N/A", 0.0
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT: [negative, neutral, positive]
        negative = predictions[0][0].item()
        neutral = predictions[0][1].item()
        positive = predictions[0][2].item()
        
        # Calculate compound score [-1, 1]
        compound = positive - negative
        
        # Determine label
        if positive > negative and positive > neutral:
            label = "Positive"
        elif negative > positive and negative > neutral:
            label = "Negative"
        else:
            label = "Neutral"
        
        return label, compound
        
    except Exception:
        return "N/A", 0.0


def analyze_news_sentiment_bert_only(news_data: list, use_bert: bool = True):
    """
    FinBERT-only sentiment analysis for news
    
    Args:
        news_data: List of news items
        use_bert: Whether to use FinBERT
    """
    # Check FinBERT availability
    bert_available = use_bert and (load_finbert_model()[0] is not None)
    
    for item in news_data:
        title = item['Title']
        
        # Get FinBERT sentiment
        if bert_available:
            bert_label, bert_score = get_finbert_sentiment(title)
            
            # Use FinBERT score directly
            combined_score = bert_score
            
            # Determine final label and emoji based on FinBERT
            if combined_score >= 0.05:
                final_label = 'Positive'
                final_emoji = '🟢'
                final_color = 'green'
            elif combined_score <= -0.05:
                final_label = 'Negative'
                final_emoji = '🔴'
                final_color = 'red'
            else:
                final_label = 'Neutral'
                final_emoji = '🟡'
                final_color = 'orange'
        else:
            # If FinBERT not available, set as N/A
            combined_score = 0.0
            final_label = 'N/A'
            final_emoji = '⚪'
            final_color = 'gray'
            bert_label = 'N/A'
            bert_score = 0.0
        
        # Add sentiment data
        item['sentiment_label'] = final_label
        item['sentiment_score'] = combined_score
        item['sentiment_emoji'] = final_emoji
        item['sentiment_color'] = final_color
        item['bert_score'] = bert_score
        item['bert_label'] = bert_label
        item['bert_available'] = bert_available
    
    return news_data
