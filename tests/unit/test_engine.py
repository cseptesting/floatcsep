import unittest

from floatcsep.infrastructure.engine import Task, TaskGraph


class DummyClass:
    def __init__(self, name):
        self.name = name

    def dummy_method(self, value):
        return value * 2


class TestTask(unittest.TestCase):

    def setUp(self):
        self.obj = DummyClass("TestObj")
        self.task = Task(instance=self.obj, method="dummy_method", value=10)

    def test_init(self):
        self.assertEqual(self.task.obj, self.obj)
        self.assertEqual(self.task.method, "dummy_method")
        self.assertEqual(self.task.kwargs["value"], 10)

    def test_sign_match(self):
        self.assertTrue(self.task.sign_match(obj=self.obj, meth="dummy_method", kw_arg=10))
        self.assertFalse(
            self.task.sign_match(obj="NonMatching", meth="dummy_method", kw_arg=10)
        )

    def test___str__(self):
        task_str = str(self.task)
        self.assertIn("TestObj", task_str)
        self.assertIn("dummy_method", task_str)
        self.assertIn("value", task_str)

    def test_run(self):
        result = self.task.run()
        self.assertEqual(result, 20)
        self.assertEqual(self.task.store, 20)

    def test___call__(self):
        result = self.task()
        self.assertEqual(result, 20)


class TestTaskGraph(unittest.TestCase):

    def setUp(self):
        self.graph = TaskGraph()
        self.obj = DummyClass("TestObj")
        self.task_a = Task(instance=self.obj, method="dummy_method", value=10)
        self.task_b = Task(instance=self.obj, method="dummy_method", value=20)

    def test_init(self):
        self.assertEqual(self.graph.ntasks, 0)
        self.assertEqual(self.graph.name, "floatcsep.infrastructure.engine.TaskGraph")

    def test_add(self):
        self.graph.add(self.task_a)
        self.assertIn(self.task_a, self.graph.tasks)
        self.assertEqual(self.graph.ntasks, 1)

    def test_add_dependency(self):
        self.graph.add(self.task_a)
        self.graph.add(self.task_b)
        self.graph.add_dependency(
            self.task_b, dep_inst=self.obj, dep_meth="dummy_method", dkw=10
        )
        self.assertIn(self.task_a, self.graph.tasks[self.task_b])

    def test_run(self):
        self.graph.add(self.task_a)
        self.graph.run()
        self.assertEqual(self.task_a.store, 20)

    def test___call__(self):
        self.graph.add(self.task_a)
        self.graph()
        self.assertEqual(self.task_a.store, 20)


if __name__ == "__main__":
    unittest.main()
