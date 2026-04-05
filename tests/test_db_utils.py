"""Unit tests for db_utils.py."""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.db_utils import (
    execute_sql,
    get_all_tables,
    get_table_info,
    get_foreign_keys,
    get_column_samples,
    get_column_stats,
    build_ddl_schema,
    build_light_schema,
    compare_results,
)


class TestDBUtils(unittest.TestCase):
    """Tests for database utility functions."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary SQLite database with 3 tables."""
        cls.tmp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmp_dir, "test.sqlite")

        conn = sqlite3.connect(cls.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY,
                dept_name TEXT NOT NULL,
                budget REAL DEFAULT 0.0
            );
        """)

        cursor.execute("""
            CREATE TABLE employees (
                emp_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dept_id INTEGER,
                salary REAL,
                hire_date TEXT,
                FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                project_name TEXT NOT NULL,
                lead_emp_id INTEGER,
                dept_id INTEGER,
                start_date TEXT,
                FOREIGN KEY (lead_emp_id) REFERENCES employees(emp_id),
                FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
            );
        """)

        # Insert data
        cursor.executemany(
            "INSERT INTO departments VALUES (?, ?, ?)",
            [
                (1, "Engineering", 500000.0),
                (2, "Marketing", 200000.0),
                (3, "Sales", 300000.0),
            ],
        )

        cursor.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Alice", 1, 120000.0, "2020-01-15"),
                (2, "Bob", 1, 110000.0, "2020-03-20"),
                (3, "Charlie", 2, 90000.0, "2021-06-01"),
                (4, "Diana", 3, 95000.0, "2019-11-10"),
                (5, "Eve", 1, 130000.0, "2018-05-22"),
                (6, "Frank", 2, 85000.0, "2022-01-01"),
                (7, "Grace", 3, 105000.0, "2020-09-15"),
                (8, "Hank", None, 70000.0, "2023-03-01"),
            ],
        )

        cursor.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Project Alpha", 1, 1, "2023-01-01"),
                (2, "Project Beta", 5, 1, "2023-06-15"),
                (3, "Campaign X", 3, 2, "2023-03-01"),
                (4, "Sales Push", 4, 3, "2023-09-01"),
            ],
        )

        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary database."""
        import shutil
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_execute_sql_select(self):
        result = execute_sql("SELECT COUNT(*) FROM employees;", self.db_path)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][0], 8)

    def test_execute_sql_with_join(self):
        result = execute_sql(
            "SELECT e.name, d.dept_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id ORDER BY e.name;",
            self.db_path,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 7)  # Hank has no dept
        self.assertEqual(result[0][0], "Alice")

    def test_execute_sql_invalid(self):
        result = execute_sql("SELECT * FROM nonexistent;", self.db_path)
        self.assertIsNone(result)

    def test_execute_sql_timeout(self):
        # This should complete quickly, just test the timeout mechanism works
        result = execute_sql("SELECT 1;", self.db_path, timeout=5)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][0], 1)

    def test_get_all_tables(self):
        tables = get_all_tables(self.db_path)
        self.assertEqual(sorted(tables), ["departments", "employees", "projects"])

    def test_get_table_info(self):
        columns = get_table_info(self.db_path, "employees")
        self.assertEqual(len(columns), 5)

        col_names = [c["name"] for c in columns]
        self.assertIn("emp_id", col_names)
        self.assertIn("name", col_names)
        self.assertIn("salary", col_names)

        # Check emp_id is primary key
        emp_id_col = next(c for c in columns if c["name"] == "emp_id")
        self.assertTrue(emp_id_col["pk"])

        # Check name is NOT NULL
        name_col = next(c for c in columns if c["name"] == "name")
        self.assertTrue(name_col["notnull"])

    def test_get_foreign_keys(self):
        fks = get_foreign_keys(self.db_path)
        self.assertTrue(len(fks) >= 3)

        # Check employees.dept_id -> departments.dept_id
        emp_dept_fk = [fk for fk in fks if fk["from_table"] == "employees" and fk["from_column"] == "dept_id"]
        self.assertTrue(len(emp_dept_fk) > 0)
        self.assertEqual(emp_dept_fk[0]["to_table"], "departments")

    def test_get_column_samples(self):
        samples = get_column_samples(self.db_path, "departments", "dept_name", n=5)
        self.assertTrue(len(samples) > 0)
        self.assertIn("Engineering", samples)

    def test_get_column_samples_empty(self):
        samples = get_column_samples(self.db_path, "nonexistent", "col", n=5)
        self.assertEqual(samples, [])

    def test_get_column_stats(self):
        stats = get_column_stats(self.db_path, "employees", "salary")
        self.assertEqual(stats["distinct_count"], 8)
        self.assertEqual(stats["min"], 70000.0)
        self.assertEqual(stats["max"], 130000.0)
        self.assertTrue(len(stats["top_values"]) > 0)

        # Check null count for dept_id (Hank has NULL)
        dept_stats = get_column_stats(self.db_path, "employees", "dept_id")
        self.assertEqual(dept_stats["null_count"], 1)

    def test_build_ddl_schema(self):
        ddl = build_ddl_schema(self.db_path)
        self.assertIn("CREATE TABLE", ddl)
        self.assertIn("departments", ddl)
        self.assertIn("employees", ddl)
        self.assertIn("projects", ddl)
        self.assertIn("PRIMARY KEY", ddl)

    def test_build_ddl_schema_with_enrichments(self):
        enrichments = {
            "departments": {
                "dept_name": {"description": "Name of the department"},
            }
        }
        ddl = build_ddl_schema(self.db_path, enrichments=enrichments)
        self.assertIn("Name of the department", ddl)

    def test_build_light_schema(self):
        schema = build_light_schema(self.db_path)
        self.assertIn("### Table: departments", schema)
        self.assertIn("### Table: employees", schema)
        self.assertIn("Column", schema)
        self.assertIn("Type", schema)

    def test_compare_results_equal(self):
        r1 = [(1, "Alice"), (2, "Bob")]
        r2 = [(2, "Bob"), (1, "Alice")]
        self.assertTrue(compare_results(r1, r2))

    def test_compare_results_different(self):
        r1 = [(1, "Alice")]
        r2 = [(1, "Bob")]
        self.assertFalse(compare_results(r1, r2))

    def test_compare_results_float_tolerance(self):
        r1 = [(1, 3.14159)]
        r2 = [(1, 3.14160)]
        self.assertTrue(compare_results(r1, r2))

    def test_compare_results_different_lengths(self):
        r1 = [(1,), (2,)]
        r2 = [(1,)]
        self.assertFalse(compare_results(r1, r2))

    def test_compare_results_both_none(self):
        self.assertTrue(compare_results(None, None))

    def test_compare_results_one_none(self):
        self.assertFalse(compare_results([(1,)], None))
        self.assertFalse(compare_results(None, [(1,)]))

    def test_compare_results_empty(self):
        self.assertTrue(compare_results([], []))

    def test_compare_results_string_case(self):
        r1 = [(1, "Alice")]
        r2 = [(1, "alice")]
        self.assertTrue(compare_results(r1, r2))


if __name__ == "__main__":
    unittest.main()
