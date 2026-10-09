from typing import List
from typing import Tuple

import spacy
from checklist.expect import Expect
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT
from model import SentimentModel


def run_negation_test(model: SentimentModel, sents: List[str]) -> MFT:
    """negation test
    """
    # Wrap the model for checklist
    wrapped_model = PredictorWrapper.wrap_predict(model.predict_proba)

    # Either add or remove negation from the sentence
    nlp = spacy.load("en_core_web_sm")
    pdata = list(nlp.pipe(sents))
    _testcases = Perturb.perturb(pdata, Perturb.remove_negation).data
    _testcases += Perturb.perturb(pdata, Perturb.add_negation).data
    testcases = []
    for sent in sents:
        for testcase in _testcases:
            if sent in testcase:
                testcases.append(testcase)
                break

    # Define the response function
    # Check documentation for more details
    # Return [True, True] if the testcase is passed
    # Return [False, False] if the testcase is failed
    def response(xs, preds, confs, labels=None, meta=None):
        """test case func
        """
        res = abs(preds[0][-1] - preds[1][-1])
        if res > 0.3:
            res = [True, True]
        else:
            res = [False, False]
        return res

    # Create and run the test
    # Expect should work with testcases
    test = MFT(testcases,
               name='negation test',
               description='the sents should have opposite label',
               expect=Expect.testcase(response))
    test.run(wrapped_model, overwrite=True)

    return test


if __name__ == "__main__":
    model = SentimentModel()
    sents = [
        "The delivery was swift and on time.",
        "I wasn't disappointed with the service.",
        "The food arrived cold and unappetizing.",
        "Their app is quite user-friendly and intuitive.",
        "I didn't find their selection lacking.",
        "The delivery person was rude and impatient.",
        "They always have great deals and offers.",
        "I haven't had any bad experiences yet.",
        "I was amazed by the quick response to my complaint.",
        "Their tracking system isn't always accurate.",
    ]

    test = run_negation_test(model, sents)

    def format_example(x, pred, conf, label=None, meta=None):
        #return f"{x} (pos. class conf.: {conf[2]:.2f})"
        return f"{x} (pos. class conf.: {pred[2]:.2f})"

    print(test.summary(n=5, format_example_fn=format_example))
