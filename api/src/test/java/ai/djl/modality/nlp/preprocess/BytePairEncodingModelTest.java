package ai.djl.modality.nlp.preprocess;

import java.util.ArrayList;
import java.util.List;
import ai.djl.modality.nlp.preprocess.AbstractSentencePieceModel.SentencePiece;
import ai.djl.modality.nlp.preprocess.AbstractSentencePieceModel.SentencePieceType;
import ai.djl.modality.nlp.preprocess.AbstractSentencePieceModel.EncodeResult;
import org.testng.annotations.Test;

import static ai.djl.modality.nlp.preprocess.AbstractSentencePieceModel.SentencePieceType.NORMAL;
import static ai.djl.modality.nlp.preprocess.AbstractSentencePieceModel.SentencePieceType.UNUSED;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.assertEquals;

public class BytePairEncodingModelTest {

    private void addPiece(List<SentencePiece> pieces,
                          String piece, float score, SentencePieceType type)
    {
        SentencePiece newPiece = new SentencePiece();
        newPiece.piece = piece;
        newPiece.score = score;
        newPiece.type  = type;
        pieces.add(newPiece);
    }

    private void addPiece(List<SentencePiece> pieces,
                          String piece, SentencePieceType type)
    {
        addPiece(pieces, piece, 0f, type);
    }

    private List<SentencePiece> makeBaseModelPieces() {
        List<SentencePiece> pieces = new ArrayList<>();
        addPiece(pieces, "<unk>", SentencePieceType.UNKOWN);
        addPiece(pieces, "<s>", SentencePieceType.CONTROL);
        addPiece(pieces, "</s>", SentencePieceType.CONTROL);
        return pieces;
    }

    @Test
    public void testEncode() {
        List<SentencePiece> baseModel = makeBaseModelPieces();
        addPiece(baseModel, "ab", 0.0f, NORMAL);         // 3
        addPiece(baseModel, "cd", -0.1f, NORMAL);        // 4
        addPiece(baseModel, "abc", -0.2f, NORMAL);       // 5
        addPiece(baseModel, "a", -0.3f, NORMAL);         // 6
        addPiece(baseModel, "b", -0.4f, NORMAL);         // 7
        addPiece(baseModel, "c", -0.5f, NORMAL);         // 8
        addPiece(baseModel, "ABC", -0.5f, SentencePieceType.USER_DEFINED);       // 9
        addPiece(baseModel, "abcdabcd", -0.5f, SentencePieceType.USER_DEFINED);  // 10
        addPiece(baseModel, "q", -0.5f, SentencePieceType.USER_DEFINED);         // 11
        addPiece(baseModel, "r", -0.5f, SentencePieceType.USER_DEFINED);         // 12
        addPiece(baseModel, "qr", -0.5f, NORMAL);        // 13

        BytePairEncodingModel model = new BytePairEncodingModel(baseModel);

        EncodeResult result;

        result = model.encode("");
        assertTrue(result.isEmpty());

        result = model.encode("abc");
        assertEquals(1, result.size());
        assertEquals("abc", result.get(0).getKey());

        result = model.encode("AB");
        assertEquals(2, result.size());
        assertEquals("A", result.get(0).getKey());
        assertEquals("B", result.get(1).getKey());

        result = model.encode("abcd");
        assertEquals(2, result.size());
        assertEquals("ab", result.get(0).getKey());
        assertEquals("cd", result.get(1).getKey());

        result = model.encode("abcc");
        assertEquals(2, result.size());
        assertEquals("abc", result.get(0).getKey());
        assertEquals("c", result.get(1).getKey());

        result = model.encode("xabcabaabcdd");
        assertEquals(7, result.size());
        assertEquals("x", result.get(0).getKey());
        assertEquals("abc", result.get(1).getKey());
        assertEquals("ab", result.get(2).getKey());
        assertEquals("a", result.get(3).getKey());
        assertEquals("ab", result.get(4).getKey());
        assertEquals("cd", result.get(5).getKey());
        assertEquals("d", result.get(6).getKey());

        // all unknown.
        result = model.encode("xyz東京");
        assertEquals(5, result.size());
        assertEquals("x", result.get(0).getKey());
        assertEquals("y", result.get(1).getKey());
        assertEquals("z", result.get(2).getKey());
        assertEquals("東", result.get(3).getKey());
        assertEquals("京", result.get(4).getKey());

        // User defined
        result = model.encode("ABC");
        assertEquals(1, result.size());
        assertEquals("ABC", result.get(0).getKey());

        result = model.encode("abABCcd");
        assertEquals(3, result.size());
        assertEquals("ab", result.get(0).getKey());
        assertEquals("ABC", result.get(1).getKey());
        assertEquals("cd", result.get(2).getKey());

        // middle "abcdabcd" is user defined.
        result = model.encode("ababcdabcdcd");
        assertEquals(3, result.size());
        assertEquals("ab", result.get(0).getKey());
        assertEquals("abcdabcd", result.get(1).getKey());
        assertEquals("cd", result.get(2).getKey());

        result = model.encode("abqrcd");
        assertEquals(4, result.size());
        assertEquals("ab", result.get(0).getKey());
        assertEquals("q", result.get(1).getKey());
        assertEquals("r", result.get(2).getKey());
        assertEquals("cd", result.get(3).getKey());
    }

    @Test
    public void testEncodeAmbiguous() {
        List<SentencePiece> baseModel = makeBaseModelPieces();

        addPiece(baseModel, "aa", -0.1f, NORMAL);
        addPiece(baseModel, "bb", -0.2f, NORMAL);
        addPiece(baseModel, "ab", -0.3f, NORMAL);
        addPiece(baseModel, "a", -0.4f, NORMAL);
        addPiece(baseModel, "b", -0.5f, NORMAL);

        BytePairEncodingModel model = new BytePairEncodingModel(baseModel);

        EncodeResult result;

        // leftmost symbols are merged first.
        result = model.encode("aaa");
        assertEquals(2, result.size());
        assertEquals("aa", result.get(0).getKey());
        assertEquals("a", result.get(1).getKey());

        // "bb" is replaced earlier than "ab".
        result = model.encode("aabb");
        assertEquals(2, result.size());
        assertEquals("aa", result.get(0).getKey());
        assertEquals("bb", result.get(1).getKey());

        // "bb" is replaced earlier than "ab".
        result = model.encode("aaabbb");
        assertEquals(4, result.size());
        assertEquals("aa", result.get(0).getKey());
        assertEquals("a", result.get(1).getKey());
        assertEquals("bb", result.get(2).getKey());
        assertEquals("b", result.get(3).getKey());

        result = model.encode("aaaba");
        assertEquals(3, result.size());
        assertEquals("aa", result.get(0).getKey());
        assertEquals("ab", result.get(1).getKey());
        assertEquals("a", result.get(2).getKey());

        // makes a broken utf-16 string
        String brokenUtf16 = "\uD83D\uDCA9".substring(0, 1);
        result = model.encode(brokenUtf16);
        assertEquals(1, result.size());
        assertEquals(brokenUtf16, result.get(0).getKey());
    }

    @Test
    public void testEncodeWithUnused() {
        List<SentencePiece> baseModel = makeBaseModelPieces();

        addPiece(baseModel, "abcd", 10.0f, NORMAL);  // 3
        addPiece(baseModel, "abc", 5.0f, NORMAL);    // 4
        addPiece(baseModel, "ab", 2.0f, NORMAL);     // 5
        addPiece(baseModel, "cd", 1.0f, NORMAL);     // 6
        addPiece(baseModel, "a", 0.0f, NORMAL);      // 7
        addPiece(baseModel, "b", 0.0f, NORMAL);      // 8
        addPiece(baseModel, "c", 0.0f, NORMAL);      // 9
        addPiece(baseModel, "d", 0.0f, NORMAL);      // 10

        // No unused.
        {
            BytePairEncodingModel model = new BytePairEncodingModel(baseModel);
            EncodeResult result = model.encode("abcd");
            assertEquals(1, result.size());
            assertEquals("abcd", result.get(0).getKey());
        }

        {
            baseModel.get(3).type = UNUSED;
            BytePairEncodingModel model = new BytePairEncodingModel(baseModel);
            EncodeResult result = model.encode("abcd");
            assertEquals(2, result.size());
            assertEquals("abc", result.get(0).getKey());
            assertEquals("d", result.get(1).getKey());
        }

        {
            // The parent rule "abc" is still alive even if the child "ab" is unused.
            baseModel.get(3).type = UNUSED;
            baseModel.get(5).type = UNUSED;
            BytePairEncodingModel model = new BytePairEncodingModel(baseModel);
            EncodeResult result = model.encode("abcd");
            assertEquals(2, result.size());
            assertEquals("abc", result.get(0).getKey());
            assertEquals("d", result.get(1).getKey());
        }

        {
            // This is tricky case. Even though "cd" is alive, it is not used, as
            // it is not merged during the segmentation step.
            // Segmentation: a|b|c|d => ab|c|d| => abc|d => abcd
            // Resegmentation: abcd => abc|d => ab|c|d. ("abcd", "abc" are unsued)
            baseModel.get(3).type = UNUSED;
            baseModel.get(4).type = UNUSED;
            baseModel.get(5).type = NORMAL;
            BytePairEncodingModel model = new BytePairEncodingModel(baseModel);
            EncodeResult result = model.encode("abcd");
            assertEquals(3, result.size());
            assertEquals("ab", result.get(0).getKey());
            assertEquals("c", result.get(1).getKey());
            assertEquals("d", result.get(2).getKey());
        }
    }
}
