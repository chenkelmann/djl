package ai.djl.modality.nlp.preprocess;

import ai.djl.ndarray.NDList;
import ai.djl.translate.PreProcessor;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Base class for SentencePiece models.
 */
public abstract class AbstractSentencePieceModel
        implements PreProcessor<String>, TextProcessor, Tokenizer
{

    public enum SentencePieceType {
        NORMAL, UNKOWN, CONTROL, USER_DEFINED, UNUSED
    }

    public static class SentencePiece {
        String piece = "";
        float score = 0f;
        SentencePieceType type = SentencePieceType.NORMAL;
    }

    public static class EncodeResult extends PairList<String, Integer> {}

    /**
     * Mapping of normal pieces to their ids
     */
    protected Map<String, Integer> pieceToIdMap = new HashMap<>();
    /**
     * Mapping of reserved pieces (unknown, user defined, etc.) to their ids
     */
    protected Map<String, Integer> pieceToReservedIdMap = new HashMap<>();
    /**
     * List of all pieces, normal and reserved. Index in the list is the id of the piece.
     */
    protected List<SentencePiece> pieces;
    /**
     * the unknown piece
     */
    protected String unkPiece = "<unk>";
    /**
     * the beginning of Sequence piece
     */
    protected String bosPiece = "<s>";
    /**
     * the end of sequence piece
     */
    protected String eosPiece = "</s>";
    /**
     * the padding piece
     */
    protected String padPiece = "<pad>";
    /**
     * Id to use for the unknown token
     */
    protected int unkId = -1;
    /**
     * Matcher for user defined pieces.
     */
    protected PrefixMatcher prefixMatcher;

    public AbstractSentencePieceModel(List<SentencePiece> pieces, String unkPiece,
                                      String bosPiece, String eosPiece,
                                      String padPiece)
    {
        this.pieces = pieces;
        if (unkPiece != null && !unkPiece.isEmpty()) { this.unkPiece = unkPiece; }
        if (bosPiece != null && !bosPiece.isEmpty()) { this.bosPiece = bosPiece; }
        if (eosPiece != null && !eosPiece.isEmpty()) { this.eosPiece = eosPiece; }
        if (padPiece != null && !padPiece.isEmpty()) { this.padPiece = padPiece; }

        List<String> userDefinedSymbols = new ArrayList<>();

        for (int i = 0; i < pieces.size(); i++) {
            SentencePiece sp = pieces.get(i);

            if (sp.piece.isEmpty()) {
                throw new IllegalArgumentException("Piece must not be empty.");
            }

            // Pieces that are part of the "normal" text are considered normal in this context,
            // Control & unknown pieces are treated differently
            boolean isNormalPiece = sp.type == SentencePieceType.NORMAL ||
                                    sp.type == SentencePieceType.USER_DEFINED ||
                                    sp.type == SentencePieceType.UNUSED;
            Integer previousId =
                    (isNormalPiece ? pieceToIdMap : pieceToReservedIdMap).put(sp.piece, i);
            if (previousId != null) {
                throw new IllegalArgumentException("'" + sp.piece + "' is already defined.");
            }

            if (sp.type == SentencePieceType.USER_DEFINED) {
                userDefinedSymbols.add(sp.piece);
            }
            if (sp.type == SentencePieceType.UNKOWN) {
                if (unkId >= 0) {
                    throw new IllegalArgumentException("'unk' is already defined.");
                }
                unkId = i;
            }
        }

        if (unkId == -1) {
            throw new IllegalArgumentException("'unk' is not defined.");
        }

        this.prefixMatcher = new PrefixMatcher(userDefinedSymbols);
    }

    /** Performs tokenization according to this model.
     * @param normalized A unicode normalized non-null input string
     * @return The resulting tokenization, a list of pieces and their ids. The concatenation of the pieces must yield
     * the original string.
     */
    public abstract EncodeResult encode(final String normalized);

    @Override
    public List<String> tokenize(String sentence) {
        return encode(sentence).stream().map(Pair::getKey).collect(Collectors.toList());
    }

    @Override
    public String buildSentence(List<String> tokens) {
        return tokens.stream().collect(Collectors.joining());
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        return new NDList(ctx.getNDManager().create(
                encode(input).stream().map(Pair::getValue).mapToInt((i) -> i).toArray())
        );
    }

    /**
     * Returns the unknown piece
     * @return the unknown piece
     */
    public String getUnkPiece() { return unkPiece; }

    /**
     * Returns the begin of sequence piece
     * @return the begin of sequence piece
     */
    public String getBosPiece() { return bosPiece; }

    /**
     * Returns the end of sequence piece
     * @return the end of sequence piece
     */
    public String getEosPiece() { return eosPiece; }

    /**
     * Returns the padding piece
     * @return the padding piece
     */
    public String getPadPiece() { return padPiece; }

    /**
     * Returns the id used for unknown tokens.
     * @return The id used for unknown tokens.
     */
    public int getUnkId() { return unkId; }

    /**
     * Returns the id of the given piece or {@link AbstractSentencePieceModel#getUnkId()}
     * if the piece is not part of this model.
     * @param piece a unicode normalized non-null string piece
     * @return the id of the piece or {@link AbstractSentencePieceModel#getUnkId()}
     */
    public int getId(final String piece) {
        Integer id = pieceToReservedIdMap.get(piece);
        if (id == null) {
            id = pieceToIdMap.get(piece);
        }
        if (id == null) {
            return unkId;
        } else {
            return id;
        }
    }

    /**
     * Returns the piece with the given pieceId or throws an {@link IndexOutOfBoundsException}
     * if the pieceId does not exist.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return the piece with the given pieceId
     */
    public String idToPiece(final int pieceId) {
        return pieces.get(pieceId).piece;
    }

    /**
     * Returns the number of pieces in this model (the vocabulary size)
     * @return the number of pieces in this model
     */
    public int getVocabularySize() {
        return pieces.size();
    }

    /**
     * Returns the score for the piece with the given id or throws {@link ArrayIndexOutOfBoundsException}
     * for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return the score for the piece with the given id
     */
    public float getScore(final int pieceId) {
        return pieces.get(pieceId).score;
    }

    /**
     * Returns whether the piece with the given id is a piece with type {@link SentencePieceType#UNKOWN}
     * or throws {@link ArrayIndexOutOfBoundsException} for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return true: piece with given id is unknown piece
     */
    public boolean isUnknown(final int pieceId) {
        return pieces.get(pieceId).type == SentencePieceType.UNKOWN;
    }

    /**
     * Returns whether the piece with the given id is a piece with type {@link SentencePieceType#CONTROL}
     * or throws {@link ArrayIndexOutOfBoundsException} for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return true: piece with given id is a control piece
     */
    public boolean isControl(final int pieceId) {
        return pieces.get(pieceId).type == SentencePieceType.CONTROL;
    }

    /**
     * Returns whether the piece with the given id is a piece with type {@link SentencePieceType#UNUSED}
     * or throws {@link ArrayIndexOutOfBoundsException} for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return true: piece with given id is an unused piece
     */
    public boolean isUnused(final int pieceId) {
        return pieces.get(pieceId).type == SentencePieceType.UNUSED;
    }

    /**
     * Returns whether the piece with the given id is a piece with type {@link SentencePieceType#USER_DEFINED}
     * or throws {@link ArrayIndexOutOfBoundsException} for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return true: piece with given id is a user defined piece
     */
    public boolean isUserDefined(final int pieceId) {
        return pieces.get(pieceId).type == SentencePieceType.USER_DEFINED;
    }
}
