from botocore.exceptions import ClientError

class ComprehendDetect:
    """Encapsulates Comprehend detection functions."""
    def __init__(self, comprehend_client):
        """
        :param comprehend_client: A Boto3 Comprehend client.
        """
        self.comprehend_client = comprehend_client

    def detect_entities(self, text, language_code="en"):
        """
        Detects entities in a document. Entities can be things like people and places
        or other common terms.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of entities along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_entities(
                Text=text, LanguageCode=language_code)
            entities = response['Entities']
        except ClientError:
            raise
        else:
            return entities

    def detect_key_phrases(self, text, language_code="en"):
        """
        Detects key phrases in a document. A key phrase is typically a noun and its
        modifiers.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of key phrases along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_key_phrases(
                Text=text, LanguageCode=language_code)
            phrases = response['KeyPhrases']
        except ClientError:
            raise
        else:
            return phrases

    def detect_pii(self, text, language_code="en"):
        """
        Detects personally identifiable information (PII) in a document. PII can be
        things like names, account numbers, or addresses.

        :param text: The document to inspect.
        :param language_code: The language of the document.
        :return: The list of PII entities along with their confidence scores.
        """
        try:
            response = self.comprehend_client.detect_pii_entities(
                Text=text, LanguageCode=language_code)
            entities = response['Entities']
        except ClientError:
            raise
        else:
            return entities

    def run_pipeline(self, text):
        # run all extractions and format the return value as List[str]
        entities = self.detect_entities(text)
        phrases = self.detect_key_phrases(text)
        pii_entities = self.detect_entities(text)

        ents = [ent["Text"] for ent in entities if ent["Score"]>=0.7]
        keyphrases = [phrase["Text"] for phrase in phrases if phrase["Score"]>=0.7]
        piis = [text[ent["BeginOffset"]:ent["EndOffset"]] for ent in pii_entities if ent["Score"]>=0.7]

        res = ents + keyphrases + piis
        
        return list(set(res))