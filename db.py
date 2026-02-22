"""
Database module for model performance tracking.
Stores predictions and user feedback for post-deployment evaluation.
"""
import os
import logging
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid failure when psycopg2 not installed (e.g. in minimal test env)
_conn = None


def get_db_url() -> Optional[str]:
    """Get database URL from environment."""
    return os.environ.get("DATABASE_URL")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    url = get_db_url()
    if not url:
        yield None
        return
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(url)
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Database connection failed: %s", e)
        yield None


def init_db() -> bool:
    """Create tables if they don't exist. Returns True if successful."""
    with get_connection() as conn:
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id UUID PRIMARY KEY,
                    predicted_label VARCHAR(10) NOT NULL,
                    dog_prob FLOAT NOT NULL,
                    cat_prob FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    prediction_id UUID NOT NULL REFERENCES predictions(id),
                    actual_label VARCHAR(10) NOT NULL,
                    was_correct BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.close()
            return True
        except Exception as e:
            logger.error("Failed to init DB: %s", e)
            return False


def store_prediction(predicted_label: str, dog_prob: float, cat_prob: float) -> Optional[str]:
    """Store a prediction and return its UUID. Returns None if DB unavailable."""
    pred_id = str(uuid.uuid4())
    with get_connection() as conn:
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (id, predicted_label, dog_prob, cat_prob) VALUES (%s, %s, %s, %s)",
                (pred_id, predicted_label, dog_prob, cat_prob)
            )
            cur.close()
            return pred_id
        except Exception as e:
            logger.error("Failed to store prediction: %s", e)
            return None


def store_feedback(prediction_id: str, actual_label: str, was_correct: bool) -> bool:
    """Store user feedback for a prediction. Returns True if successful."""
    with get_connection() as conn:
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO feedback (prediction_id, actual_label, was_correct) VALUES (%s, %s, %s)",
                (prediction_id, actual_label, was_correct)
            )
            cur.close()
            return True
        except Exception as e:
            logger.error("Failed to store feedback: %s", e)
            return False


def get_feedback(feedback_id: int) -> Optional[dict]:
    """Get a single feedback by id. Returns None if not found or DB unavailable."""
    with get_connection() as conn:
        if not conn:
            return None
        try:
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """SELECT f.id, f.prediction_id, f.actual_label, f.was_correct, f.created_at,
                          p.predicted_label, p.dog_prob, p.cat_prob
                   FROM feedback f
                   JOIN predictions p ON f.prediction_id = p.id
                   WHERE f.id = %s""",
                (feedback_id,)
            )
            row = cur.fetchone()
            cur.close()
            if row:
                row["prediction_id"] = str(row["prediction_id"])
                row["created_at"] = row["created_at"].isoformat() if row["created_at"] else None
            return dict(row) if row else None
        except Exception as e:
            logger.error("Failed to get feedback: %s", e)
            return None


def get_all_feedbacks(limit: int = 100, offset: int = 0) -> list:
    """List all feedbacks with optional pagination. Returns empty list if DB unavailable."""
    with get_connection() as conn:
        if not conn:
            return []
        try:
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """SELECT f.id, f.prediction_id, f.actual_label, f.was_correct, f.created_at,
                          p.predicted_label
                   FROM feedback f
                   JOIN predictions p ON f.prediction_id = p.id
                   ORDER BY f.created_at DESC
                   LIMIT %s OFFSET %s""",
                (limit, offset)
            )
            rows = cur.fetchall()
            cur.close()
            result = []
            for r in rows:
                d = dict(r)
                d["prediction_id"] = str(d["prediction_id"])
                d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
                result.append(d)
            return result
        except Exception as e:
            logger.error("Failed to list feedbacks: %s", e)
            return []


def update_feedback(feedback_id: int, actual_label: str = None, was_correct: bool = None) -> bool:
    """Update a feedback. At least one of actual_label or was_correct must be provided."""
    if actual_label is None and was_correct is None:
        return False
    updates = []
    params = []
    if actual_label is not None:
        updates.append("actual_label = %s")
        params.append(actual_label)
    if was_correct is not None:
        updates.append("was_correct = %s")
        params.append(was_correct)
    params.append(feedback_id)
    with get_connection() as conn:
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE feedback SET {', '.join(updates)} WHERE id = %s",
                params
            )
            updated = cur.rowcount > 0
            cur.close()
            return updated
        except Exception as e:
            logger.error("Failed to update feedback: %s", e)
            return False


def delete_feedback(feedback_id: int) -> bool:
    """Delete a feedback by id. Returns True if deleted, False otherwise."""
    with get_connection() as conn:
        if not conn:
            return False
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM feedback WHERE id = %s", (feedback_id,))
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        except Exception as e:
            logger.error("Failed to delete feedback: %s", e)
            return False


def get_model_metrics() -> dict:
    """
    Compute model performance metrics from stored feedback.
    Returns dict with: total_feedback, correct, incorrect, accuracy,
    tp_cat, fp_cat, fn_cat, precision_cat, recall_cat,
    tp_dog, fp_dog, fn_dog, precision_dog, recall_dog.
    """
    with get_connection() as conn:
        if not conn:
            return {}
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT p.predicted_label, f.actual_label, f.was_correct
                FROM feedback f
                JOIN predictions p ON f.prediction_id = p.id
            """)
            rows = cur.fetchall()
            cur.close()

            total = len(rows)
            correct = sum(1 for r in rows if r[2])
            incorrect = total - correct
            accuracy = correct / total if total > 0 else 0.0

            # Confusion matrix for Cat (positive class)
            tp_cat = sum(1 for r in rows if r[0] == "Cat" and r[1] == "Cat")
            fp_cat = sum(1 for r in rows if r[0] == "Cat" and r[1] == "Dog")
            fn_cat = sum(1 for r in rows if r[0] == "Dog" and r[1] == "Cat")
            tn_cat = sum(1 for r in rows if r[0] == "Dog" and r[1] == "Dog")

            precision_cat = tp_cat / (tp_cat + fp_cat) if (tp_cat + fp_cat) > 0 else 0.0
            recall_cat = tp_cat / (tp_cat + fn_cat) if (tp_cat + fn_cat) > 0 else 0.0

            # Confusion matrix for Dog
            tp_dog = tn_cat  # Dog as positive: TP = predicted Dog, actual Dog
            fp_dog = fn_cat  # predicted Dog, actual Cat
            fn_dog = fp_cat  # predicted Cat, actual Dog

            precision_dog = tp_dog / (tp_dog + fp_dog) if (tp_dog + fp_dog) > 0 else 0.0
            recall_dog = tp_dog / (tp_dog + fn_dog) if (tp_dog + fn_dog) > 0 else 0.0

            return {
                "total_feedback": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": accuracy,
                "tp_cat": tp_cat,
                "fp_cat": fp_cat,
                "fn_cat": fn_cat,
                "precision_cat": precision_cat,
                "recall_cat": recall_cat,
                "tp_dog": tp_dog,
                "fp_dog": fp_dog,
                "fn_dog": fn_dog,
                "precision_dog": precision_dog,
                "recall_dog": recall_dog,
            }
        except Exception as e:
            logger.error("Failed to get metrics: %s", e)
            return {}
