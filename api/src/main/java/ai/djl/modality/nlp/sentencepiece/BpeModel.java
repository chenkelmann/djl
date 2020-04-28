package ai.djl.modality.nlp.sentencepiece;

import ai.djl.util.Pair;
import ai.djl.util.PairList;
import sun.jvm.hotspot.debugger.cdbg.Sym;

import java.util.*;
import java.util.function.BiConsumer;

public class BpeModel {

    /**
     * A pair of symbols of this model.
     */
    private static class SymbolPair implements Comparable<SymbolPair> {
        /** left index of this pair */
        int left;
        /** right index of this pair */
        int right;
        /** score of this pair. large is better. */
        float score;
        /** length of this piece */
        int size;

        /**
         * Compares two {@link SymbolPair} instances, first by score, then by left position.
         */
        @Override
        public int compareTo(final SymbolPair that) {
            final int scoreComparison = Float.compare(this.score, that.score);
            if (scoreComparison != 0) { return scoreComparison; }
            return Integer.compare(this.left, that.left);
        }
    }

    private static class Symbol {
        /** prev index of this symbol. -1 for BOS. */
        int prev;
        /** next index of this symbol. -1 for EOS. */
        int next;
        /** this symbol is never be merged. */
        boolean freeze;
        /** the string piece this symbol represents */
        String piece;
    } ;

    /**
    Model(const ModelProto &model_proto) {
        model_proto_ = &model_proto;
        InitializePieces();
    }**/

    public BpeModel() {
    }

    /**
     * Applies this BPE Model to the given string.
     * @param normalized A unicode normalized non-null string. Most models require
     *                   {@link java.text.Normalizer.Form#NFKC} normalization.
     * @return A list of pairs consisting of the tokens and their respective ids, never null
     */
    public PairList<String, Integer> encode(final String normalized) {
        if (normalized.isEmpty()) { return new PairList<>(); }
        //TODO: check for initialization

        final PriorityQueue<SymbolPair> agenda = new PriorityQueue<>();
        final ArrayList<Symbol> symbols = new ArrayList<>(normalized.length());
        // Reverse merge rules.
        // key: merged symbol, value: pair of original symbols.
        final Map<String, Pair<String, String>> revMerge = new HashMap<>();

        // Pre-allocates SymbolPair for efficiency.
        constexpr size_t kPreallocateSymbolPairSize = 256;
        model::FreeList < SymbolPair > symbol_pair_allocator(kPreallocateSymbolPairSize);

        // Lookup new symbol pair at [left, right] and inserts it to agenda.
        final BiConsumer<Integer, Integer> maybeAddNewSymbolPair = (final Integer left, final Integer right) -> {
            // we have reached BOS/EOS or the respective symbols shoult not be split further
            if (left == -1 || right == -1 || symbols.get(left).freeze || symbols.get(right).freeze) { return; }
            const absl::string_view piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
            const auto it = pieces_.find(piece);
            if (it == pieces_.end()) {
                return;
            }
            final SymbolPair h = new SymbolPair();
            h.left = left;
            h.right = right;
            h.score = GetScore(it -> second);
            h.size = piece.size();
            agenda.add(h);

            // Makes `rev_merge` for resegmentation.
            if (IsUnusedInlined(it -> second)) {
                rev_merge[piece] =
                        std::make_pair (symbols[left].piece, symbols[right].piece);
            }
        };

        // Splits the input into character sequence
        int index = 0;
        while (!normalized.empty()) {
            Symbol s;
            const int mblen = matcher_ -> PrefixMatch(normalized, & s.freeze);
            s.piece = absl::string_view (normalized.data(), mblen);
            s.prev = index == 0 ? -1 : index - 1;
            normalized.remove_prefix(mblen);
            s.next = normalized.empty() ? -1 : index + 1;
            ++index;
            symbols.emplace_back(s);
        }

        if (symbols.empty()) {
            return {};
        }

        // Lookup all bigrams.
        for (size_t i = 1; i < symbols.size(); ++i) {
            MaybeAddNewSymbolPair(i - 1, i);
        }

        // Main loop.
        while (!agenda.empty()) {
            SymbolPair * top = agenda.top();
            agenda.pop();

            // `top` is no longer available.
            if (symbols[top -> left].piece.empty() || symbols[top -> right].piece.empty() ||
                    symbols[top -> left].piece.size() + symbols[top -> right].piece.size() !=
                            top -> size) {
                continue;
            }

            // Replaces symbols with `top` rule.
            symbols[top -> left].piece = absl::string_view (
                    symbols[top -> left].piece.data(),
                    symbols[top -> left].piece.size() + symbols[top -> right].piece.size());

            // Updates prev/next pointers.
            symbols[top -> left].next = symbols[top -> right].next;
            if (symbols[top -> right].next >= 0) {
                symbols[symbols[top -> right].next].prev = top -> left;
            }
            symbols[top -> right].piece = absl::string_view ("");

            // Adds new symbol pairs which are newly added after symbol replacement.
            MaybeAddNewSymbolPair(symbols[top -> left].prev, top -> left);
            MaybeAddNewSymbolPair(top -> left, symbols[top -> left].next);
        }

        std::function <void(absl::string_view, EncodeResult *)>resegment;
        resegment = [this, &resegment, &rev_merge](absl::string_view w, EncodeResult * output) ->void {
            const int id = PieceToId(w);
            if (id == -1 || !IsUnusedInlined(id)) {
                output -> emplace_back(w, id);
                return;
            }
            const auto p = rev_merge.find(w);
            if (p == rev_merge.end()) {
                // This block will never be called, as `rev_merge` stores all the
                // resegmentation info for unused id.
                output -> emplace_back(w, id);
                return;
            }
            // Recursively resegment left and right symbols.
            resegment(p -> second.first, output);
            resegment(p -> second.second, output);
        } ;

        EncodeResult output;
        for (int index = 0; index != -1; index = symbols[index].next) {
            CHECK_GE(index, 0);
            CHECK_LT(index, static_cast < int>(symbols.size()));
            resegment(symbols[index].piece, & output);
        }

        return output;
    }
}