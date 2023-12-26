"""
Defines POS tag encoder for predefined coding schemes.
"""
from typing import List


class POSTagsEncoder(object):
    """Handles creation of Pos tags for a list of named entity classes and
    conversion of tags to ids and vice versa."""

    def __init__(self,
                    classes: List[str],
                    ignore_index: int = 0):
                    # ignore_index: int = -100):

        if not len(set(classes)) == len(classes):
            raise ValueError("`classes` have duplicate entries.")
        # if ignore_index >= 0 or not isinstance(ignore_index, int):
        #     raise ValueError("`ignore_index` should be a negative int.")

        self.classes = tuple(classes)
        self.pos_tags = ["O"]
        self.ignore_index = ignore_index
        self.pos_tag_to_id = {"X": ignore_index}
        
        for clss in classes:
            self.pos_tags.append(f"{clss}")

        for i, tag in enumerate(self.pos_tags):
            self.pos_tag_to_id[tag] = i

    def __repr__(self):
        return ('{class_}(classes={classes!r}') \
            .format(class_=self.__class__.__name__,
                    classes=self.classes)

    @classmethod
    def from_labels_file(cls, filepath: str, *args, **kwargs):
        """Creates encoder from a file with POS label classes (one class per
        line) and a given scheme."""
        with open(filepath, 'r') as fd:
            pos_classes = [clss for clss in fd.read().splitlines() if clss]

        return cls(pos_classes, *args, **kwargs)

    @property
    def num_labels(self) -> int:
        return len(self.pos_tags)

    def convert_tags_to_ids(self, pos_tags: List[str]) -> List[int]:
        """Converts a list of tag strings to a list of tag ids."""
        return [self.pos_tag_to_id[pos_tag] for pos_tag in pos_tags]

    def convert_ids_to_tags(self, tag_ids: List[int]) -> List[str]:
        """Returns a list of tag strings from a list of tag ids."""
        return [self.pos_tags[tag_id] for tag_id in tag_ids]

    def decode_valid(self, tag_sequence: List[str]) -> List[str]:
        """Processes a list of tag strings to remove invalid predictions."""

        final = []
        for tag_and_cls in tag_sequence:
            # Return alwayws valida pos tag (TODO: Verify)
            valid_tag =  True

            if valid_tag:
                final.append(tag_and_cls)
            else:
                final.append('O')

        return final