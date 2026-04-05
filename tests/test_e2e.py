"""End-to-end test for the bird-text2sql pipeline."""
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestE2E(unittest.TestCase):
    """End-to-end pipeline test with minimal fake data."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary BIRD-like dataset structure."""
        cls.tmp_dir = tempfile.mkdtemp()
        cls.project_root = Path(cls.tmp_dir)

        # Create directory structure
        train_dir = cls.project_root / "data" / "raw" / "train"
        dev_dir = cls.project_root / "data" / "raw" / "dev"
        train_db_dir = train_dir / "train_databases" / "test_db"
        dev_db_dir = dev_dir / "dev_databases" / "test_db"

        for d in [train_dir, dev_dir, train_db_dir, dev_db_dir,
                  cls.project_root / "data" / "clean",
                  cls.project_root / "data" / "multitask",
                  cls.project_root / "data" / "schemas",
                  cls.project_root / "data" / "cache",
                  cls.project_root / "models" / "sft",
                  cls.project_root / "models" / "rl",
                  cls.project_root / "models" / "final",
                  cls.project_root / "logs"]:
            d.mkdir(parents=True, exist_ok=True)

        # Create SQLite database
        db_path = train_db_dir / "test_db.sqlite"
        cls.db_path = db_path
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE students (
                student_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                grade TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY,
                course_name TEXT NOT NULL,
                credits INTEGER
            );
        """)
        cursor.execute("""
            CREATE TABLE enrollments (
                enrollment_id INTEGER PRIMARY KEY,
                student_id INTEGER,
                course_id INTEGER,
                score REAL,
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                FOREIGN KEY (course_id) REFERENCES courses(course_id)
            );
        """)

        # Insert data
        cursor.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", [
            (1, "Alice", 20, "A"), (2, "Bob", 21, "B"), (3, "Charlie", 22, "A"),
            (4, "Diana", 20, "C"), (5, "Eve", 23, "B"),
        ])
        cursor.executemany("INSERT INTO courses VALUES (?, ?, ?)", [
            (1, "Math", 3), (2, "Science", 4), (3, "History", 3),
        ])
        cursor.executemany("INSERT INTO enrollments VALUES (?, ?, ?, ?)", [
            (1, 1, 1, 95.0), (2, 1, 2, 88.0), (3, 2, 1, 75.0),
            (4, 2, 3, 82.0), (5, 3, 2, 91.0), (6, 4, 1, 68.0),
            (7, 4, 3, 72.0), (8, 5, 2, 85.0),
        ])

        conn.commit()
        conn.close()

        # Copy DB to dev too
        shutil.copy(str(db_path), str(dev_db_dir / "test_db.sqlite"))

        # Create train.json
        train_samples = [
            {
                "question": "How many students are there?",
                "SQL": "SELECT COUNT(*) FROM students;",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
            {
                "question": "What is the average score in Math?",
                "SQL": "SELECT AVG(score) FROM enrollments e JOIN courses c ON e.course_id = c.course_id WHERE c.course_name = 'Math';",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
            {
                "question": "List all students enrolled in Science.",
                "SQL": "SELECT s.name FROM students s JOIN enrollments e ON s.student_id = e.student_id JOIN courses c ON e.course_id = c.course_id WHERE c.course_name = 'Science';",
                "db_id": "test_db",
                "difficulty": "moderate",
                "evidence": "",
            },
            {
                "question": "Which student has the highest score?",
                "SQL": "SELECT s.name FROM students s JOIN enrollments e ON s.student_id = e.student_id ORDER BY e.score DESC LIMIT 1;",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
            {
                "question": "How many courses does each student take?",
                "SQL": "SELECT s.name, COUNT(e.course_id) FROM students s JOIN enrollments e ON s.student_id = e.student_id GROUP BY s.student_id;",
                "db_id": "test_db",
                "difficulty": "moderate",
                "evidence": "",
            },
            {
                "question": "What is the total credits for Alice?",
                "SQL": "SELECT SUM(c.credits) FROM courses c JOIN enrollments e ON c.course_id = e.course_id JOIN students s ON e.student_id = s.student_id WHERE s.name = 'Alice';",
                "db_id": "test_db",
                "difficulty": "moderate",
                "evidence": "",
            },
            {
                "question": "List students with grade A.",
                "SQL": "SELECT name FROM students WHERE grade = 'A';",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
            {
                "question": "What courses have more than 3 credits?",
                "SQL": "SELECT course_name FROM courses WHERE credits > 3;",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
            {
                "question": "Find the student with the lowest score in any course.",
                "SQL": "SELECT s.name, MIN(e.score) FROM students s JOIN enrollments e ON s.student_id = e.student_id;",
                "db_id": "test_db",
                "difficulty": "moderate",
                "evidence": "",
            },
            {
                "question": "How many enrollments are there in total?",
                "SQL": "SELECT COUNT(*) FROM enrollments;",
                "db_id": "test_db",
                "difficulty": "simple",
                "evidence": "",
            },
        ]

        with open(train_dir / "train.json", "w") as f:
            json.dump(train_samples, f)

        # Create dev.json (use 5 samples)
        with open(dev_dir / "dev.json", "w") as f:
            json.dump(train_samples[:5], f)

        # Create config
        cls.config = {
            "model": {
                "name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",  # Tiny model for testing
                "max_seq_length": 512,
                "torch_dtype": "float32",
                "load_in_4bit": False,
                "trust_remote_code": True,
                "attn_implementation": "eager",
            },
            "data": {
                "bird_train_path": str(train_dir),
                "bird_dev_path": str(dev_dir),
                "db_base_path": str(cls.project_root / "data" / "raw"),
                "clean_dir": str(cls.project_root / "data" / "clean"),
                "multitask_dir": str(cls.project_root / "data" / "multitask"),
                "schema_dir": str(cls.project_root / "data" / "schemas"),
                "cache_dir": str(cls.project_root / "data" / "cache"),
                "execution_timeout": 10,
                "max_workers": 2,
                "semantic_validation": False,
                "semantic_confidence_threshold": 0.7,
                "min_samples_after_cleaning": 1,
                "schema_format": "ddl",
                "include_tasks": ["text2sql"],
                "cot_join_threshold": 2,
                "cot_subquery": True,
                "error_types": ["wrong_aggregation"],
                "max_correction_attempts": 2,
                "deduplication": True,
            },
            "training": {
                "seed": 42,
                "output_dir": str(cls.project_root / "models" / "sft"),
                "log_dir": str(cls.project_root / "logs"),
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"],
                "num_epochs": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "max_grad_norm": 1.0,
                "gradient_checkpointing": False,
                "bf16": False,
                "tf32": False,
                "dataloader_num_workers": 0,
                "save_strategy": "steps",
                "save_steps": 999999,
                "save_total_limit": 1,
                "eval_steps": 999999,
                "eval_samples": 2,
                "wandb_project": None,
                "wandb_run_name": None,
            },
            "rl": {
                "output_dir": str(cls.project_root / "models" / "rl"),
                "num_epochs": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-6,
                "num_candidates": 2,
                "num_rollouts": 2,
                "max_new_tokens": 64,
                "temperature": 0.8,
                "kl_coeff": 0.05,
                "reward_correct": 1.0,
                "reward_executable": 0.1,
                "reward_error": 0.0,
                "reward_table_bonus": 0.05,
                "lora_rank": 8,
                "lora_alpha": 16,
                "collapse_threshold": 0.1,
                "wandb_project": None,
            },
            "inference": {
                "model_path": str(cls.project_root / "models" / "final"),
                "max_new_tokens": 64,
                "temperature": 0.1,
                "top_p": 0.95,
                "num_candidates": 2,
                "num_examples": 1,
                "icl_styles": ["direct"],
                "max_refinement_rounds": 1,
                "fix_syntax_errors": True,
                "fix_semantic_errors": False,
                "selection_method": "self_consistency",
                "chroma_persist_dir": str(cls.project_root / "data" / "cache" / "chroma"),
                "batch_size": 2,
            },
            "evaluation": {
                "dev_path": str(dev_dir),
                "db_base_path": str(cls.project_root / "data" / "raw"),
                "output_dir": str(cls.project_root / "evaluation"),
                "execution_timeout": 10,
                "difficulties": ["simple", "moderate", "challenging"],
            },
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_db_utils(self):
        """Test basic db_utils functions on our test database."""
        from scripts.db_utils import execute_sql, get_all_tables, compare_results

        tables = get_all_tables(str(self.db_path))
        self.assertEqual(len(tables), 3)

        result = execute_sql("SELECT COUNT(*) FROM students;", str(self.db_path))
        self.assertIsNotNone(result)
        self.assertEqual(result[0][0], 5)

    def test_02_data_cleaning(self):
        """Test data cleaning pipeline."""
        from scripts.data_cleaning import DataCleaner

        cleaner = DataCleaner(self.config)
        clean_data = cleaner.clean()

        self.assertIsNotNone(clean_data)
        self.assertTrue(len(clean_data) > 0)

        # Check checkpoint was saved
        clean_dir = Path(self.config["data"]["clean_dir"])
        checkpoints = list(clean_dir.glob("*checkpoint*"))
        self.assertTrue(len(checkpoints) > 0)

    def test_03_dataset_building(self):
        """Test multi-task dataset building."""
        from scripts.dataset_builder import MultitaskDatasetBuilder

        # First ensure we have clean data
        from scripts.data_cleaning import DataCleaner
        cleaner = DataCleaner(self.config)
        cleaner.clean()

        builder = MultitaskDatasetBuilder(self.config)
        builder.build()

        # Check output files
        multitask_dir = Path(self.config["data"]["multitask_dir"])
        train_file = multitask_dir / "train.jsonl"
        self.assertTrue(train_file.exists())

        from scripts.utils import load_jsonl
        train_data = load_jsonl(train_file)
        self.assertTrue(len(train_data) > 0)

        # Check format
        sample = train_data[0]
        self.assertIn("messages", sample)
        self.assertTrue(len(sample["messages"]) >= 2)

    def test_04_execution_accuracy(self):
        """Test execution accuracy computation."""
        from scripts.utils import compute_execution_accuracy

        predictions = [
            "SELECT COUNT(*) FROM students;",
            "SELECT name FROM students WHERE grade = 'A';",
        ]
        gold_sqls = [
            "SELECT COUNT(*) FROM students;",
            "SELECT name FROM students WHERE grade = 'A';",
        ]
        db_paths = [str(self.db_path), str(self.db_path)]

        acc = compute_execution_accuracy(predictions, gold_sqls, db_paths)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertEqual(acc, 1.0)  # Identical queries should match

    def test_05_evaluator(self):
        """Test the evaluator."""
        from evaluation.evaluator import BIRDEvaluator

        evaluator = BIRDEvaluator(self.config)

        predictions = [
            {"sql": "SELECT COUNT(*) FROM students;", "db_id": "test_db"},
            {"sql": "SELECT name FROM students;", "db_id": "test_db"},
        ]
        gold = [
            {"SQL": "SELECT COUNT(*) FROM students;", "db_id": "test_db", "difficulty": "simple", "question": "How many students?"},
            {"SQL": "SELECT name FROM students WHERE grade = 'A';", "db_id": "test_db", "difficulty": "simple", "question": "List grade A students"},
        ]

        results = evaluator.execution_accuracy(predictions, gold)

        self.assertIn("overall", results)
        self.assertIn("by_difficulty", results)
        acc = results["overall"]["accuracy"]
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_06_extract_sql(self):
        """Test SQL extraction from various formats."""
        from scripts.utils import extract_sql_from_text

        # SQL in code block
        text1 = "Here is the query:\n```sql\nSELECT * FROM students;\n```"
        self.assertIn("SELECT", extract_sql_from_text(text1))

        # Plain SQL
        text2 = "SELECT COUNT(*) FROM students;"
        self.assertEqual(extract_sql_from_text(text2), "SELECT COUNT(*) FROM students;")

        # SQL with explanation
        text3 = "The answer is:\nSQL: SELECT name FROM students WHERE age > 20;"
        result = extract_sql_from_text(text3)
        self.assertIn("SELECT", result)


if __name__ == "__main__":
    unittest.main()
