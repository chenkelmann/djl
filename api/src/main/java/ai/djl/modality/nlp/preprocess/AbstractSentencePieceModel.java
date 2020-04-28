package ai.djl.modality.nlp.preprocess;

import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for SentencePiece models. TGiven a normalized string, returns a sequence of sentence pieces with ids.
 */
public abstract class AbstractSentencePieceModel {

    public static final int UNK_ID = 0;



    public static enum SentencePieceType {
        NORMAL, UNKONW, CONTROL, USER_DEFINED, UNUSED
    }

    public static class SentencePiece {
        String piece = "";
        float score = 0f;
        SentencePieceType type = SentencePieceType.NORMAL;
    }

    public static class EncodeResult extends PairList<String, Integer> {}
    public static class NBestEncodeResult extends PairList<EncodeResult, Float> {}

    protected final Map<String, Integer> pieceToIdMap = new HashMap<>();

    protected final ArrayList<SentencePiece> pieces = new ArrayList<>();

    public AbstractSentencePieceModel() {
        pieces_.clear();
        reserved_id_map_.clear();
        unk_id_ = -1;

        std::set<absl::string_view> user_defined_symbols;

        for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
            if (sp.piece().empty()) {
                status_ = util::InternalError("piece must not be empty.");
                return;
            }

    const bool is_normal_piece =
                    (sp.type() == ModelProto::SentencePiece::NORMAL ||
                            sp.type() == ModelProto::SentencePiece::USER_DEFINED ||
                    sp.type() == ModelProto::SentencePiece::UNUSED);
            if (!port::InsertIfNotPresent(
                    is_normal_piece ? &pieces_ : &reserved_id_map_, sp.piece(), i)) {
                status_ = util::InternalError(sp.piece() + " is already defined.");
                return;
            }

            if (sp.type() == ModelProto::SentencePiece::USER_DEFINED) {
                user_defined_symbols.insert(sp.piece());
            }

            if (sp.type() == ModelProto::SentencePiece::UNKNOWN) {
                if (unk_id_ >= 0) {
                    status_ = util::InternalError("unk is already defined.");
                    return;
                }
                unk_id_ = i;
            }
        }

        if (unk_id_ == -1) {
            status_ = util::InternalError("unk is not defined.");
            return;
        }

        matcher_ = port::MakeUnique<normalizer::PrefixMatcher>(user_defined_symbols);
    }

    //TODO: add replacement for ModelProto, save it internally

    //virtual const ModelProto &model_proto() const { return *model_proto_; }

    //virtual const normalizer::PrefixMatcher *prefix_matcher() const { return matcher_.get(); }

    /** Performs tokenization according to this model.
     * @param normalized A unicode normalized non-null input string
     * @return The resulting tokenization, a list of pieces and their ids. The concatenation of the pieces must yield
     * the original string.
     */
    public abstract EncodeResult encode(final String normalized);

    /**
     * Same as {@link AbstractSentencePieceModel#encode(String)}, but return the {@code bestCount} number of best
     * results, not just the best one.
     * @param normalized A unicode normalized non-null input string
     * @param bestCount the number of best results to return
     * @return The resulting tokenization, a list of list of pieces and their ids.
     *         The concatenation of the pieces in each list must yield the original string.
     */
    public abstract NBestEncodeResult encode(final String normalized, final int bestCount);

    public abstract EncodeResult sampleEncode(final String normalized, final float alpha);

    /**
     * Returns the unknown piece
     * @return the unknown piece
     */
    public String unkPiece() { return "<unk>"; }

    /**
     * Returns the begin of sequence piece
     * @return the begin of sequence piece
     */
    public String bosPiece() { return "<s>"; }

    /**
     * Returns the end of sequence piece
     * @return the end of sequence piece
     */
    public String eosPiece() { return "</s>"; }

    /**
     * Returns the padding piece
     * @return the padding piece
     */
    public String padPiece() { return "<pad>"; }

    /**
     * Returns the id of the given piece or {@link AbstractSentencePieceModel#UNK_ID}
     * if the piece is not part of this model.
     * @param piece a unicode normalized non-null string piece
     * @return the id of the piece or {@link AbstractSentencePieceModel#UNK_ID}
     */
    public int pieceToId(final String piece) {
        final Integer id = pieceToIdMap.get(piece);
        if (id == null) { return UNK_ID; } else { return id; }
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
     * Returns whether the piece with the given id is a piece with type {@link SentencePieceType#UNKONW}
     * or throws {@link ArrayIndexOutOfBoundsException} for invalid ids.
     * @param pieceId the pieceId of the piece, >= 0, < {@link AbstractSentencePieceModel#getVocabularySize()}
     * @return true: piece with given id is unknown piece
     */
    public boolean isUnknown(final int pieceId) {
        return pieces.get(pieceId).type == SentencePieceType.UNKONW;
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

    protected abstract void initializePieces();

    // PrefixMatcher for user defined symbols.
    // TODO
    //std::unique_ptr<normalizer::PrefixMatcher> matcher_;
}
