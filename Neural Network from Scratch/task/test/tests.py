import numpy as np
from hstest import StageTest, TestCase, CheckResult, PlottingTest
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform

np.random.uniform = custom_uniform

# The source data I will test on

true_acc_history_res = [0.8057, 0.8309, 0.8447, 0.8501, 0.854, 0.8576, 0.8616, 0.8635, 0.8658, 0.8671, 0.8692, 0.8703,
                        0.872, 0.8735, 0.8745, 0.8758, 0.8763, 0.8769, 0.8786, 0.8794]


class Tests7(PlottingTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected lists were found in output')

        # Getting the student's results from the reply

        try:
            acc_history_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that accuracy log output is in wrong format')

        check_result = full_check(acc_history_res, true_acc_history_res, 'train sequence')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests7().run_tests()
