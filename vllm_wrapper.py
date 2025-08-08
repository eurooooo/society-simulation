from vllm import LLM, SamplingParams

class BatchedLLM(LLM):
    """
    A subclass of vllm.LLM that overwrites the `generate` method
    to process multiple prompts in smaller batches.
    """

    def __init__(self, *args, batch_size=8, **kwargs):
        """
        Args:
            batch_size (int): Maximum number of prompts to process at once.
            *args: Passed through to the base LLM class.
            **kwargs: Passed through to the base LLM class.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def generate(self, prompts, sampling_params, testing=False):
        """
        Overwrites the default `generate` method to handle lists of prompts in batches.

        Args:
            prompts (str or List[str]): The prompt(s) to generate responses for.
            sampling_params (SamplingParams): Sampling parameters.

        Returns:
            List[str] or str: A list of generated responses if multiple prompts,
                              or a single response if a single prompt.
        """

        # If it's just a single prompt (string), call parent directly.
        if isinstance(prompts, str):
            if testing:  # skip llm invoke
                return "[TEST_RESPONSE]"
            else:
                return super().generate(prompts, sampling_params)

        # Otherwise, handle a list of prompts in batches.
        all_responses = []
        total_prompts = len(prompts)

        if testing:  # skip llm invoke
            return ["[TEST_RESPONSE]" for _ in prompts]

        for i in range(0, total_prompts, self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            # Use the parent class's generate method for each batch
            batch_responses = super().generate(batch_prompts, sampling_params)
            all_responses.extend(batch_responses)

        return all_responses