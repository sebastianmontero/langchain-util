import unittest
from typing import List
from unittest.mock import Mock
from langchain.chains.base import Chain
from langchain_util.chains import ConcatenateChain

class TestConcatenateChain(unittest.TestCase):
    def test_concatenate_chain(self):

        class MockChain(Chain):
            def _call(self, inputs):
                return {
                    "output_key1": "value1",
                    "output_key2": "value2",
                }
            @property
            def input_keys(self) -> List[str]:
                return ["input_key1", "input_key2"]

            @property
            def output_keys(self) -> List[str]:
                return ["output_key1", "output_key2"]
        
        input_chain = MockChain()
        concatenate_chain = ConcatenateChain(
            input_chain=input_chain,
            keys=["output_key1"],
            output_key="output"
        )

        assert concatenate_chain.input_keys == ["input_key1", "input_key2"]
        assert concatenate_chain.output_keys == ["output"]

        result = concatenate_chain({"input_key1": "value1", "input_key2": "value2"})
        assert result == {'input_key1': 'value1', 'input_key2': 'value2', 'output': 'output_key1: value1'}

        result = concatenate_chain.run({"input_key1": "value1", "input_key2": "value2"})
        assert result == "output_key1: value1"        

        concatenate_chain = ConcatenateChain(
            input_chain=input_chain,
            keys=None,
            output_key="output"
        )
        result = concatenate_chain({"input_key1": "value1", "input_key2": "value2"})
        assert result == {'input_key1': 'value1', 'input_key2': 'value2', 'output': 'output_key1: value1 output_key2: value2'}

        result = concatenate_chain.run({"input_key1": "value1", "input_key2": "value2"})
        assert result == "output_key1: value1 output_key2: value2"
