package ai.djl.modality.nlp.preprocess;

import ai.djl.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;

public class BytePairEncodingModel extends AbstractSentencePieceModel {

    public BytePairEncodingModel(List<SentencePiece> pieces,
                                 String unkPiece,
                                 String bosPiece, String eosPiece,
                                 String padPiece)
    {
        super(pieces, unkPiece, bosPiece, eosPiece, padPiece);
    }

    public BytePairEncodingModel(List<SentencePiece> pieces) {
        this(pieces, null, null, null, null);
    }

    @Override
    public EncodeResult encode(String normalized) {
        if (normalized.isEmpty()) {
            return new EncodeResult();
        } else {
            return new BytePairEncoder(normalized).encode();
        }
    }

    private static class SymbolPair implements Comparable<SymbolPair>{
        int left;     // left index of this pair
        int right;    // right index of this pair
        float score;  // score of this pair. large is better.
        String piece; // the piece build from the concatenated symbols

        @Override
        public int compareTo(SymbolPair that) {
            // The java PriorityQueue returns the *lowest* comparable first,
            // so we need to invert the result.
            int comparison = Float.compare(this.score, that.score) * -1;
            if (comparison != 0) { return comparison; }
            return Integer.compare(this.left, that.left) * -1;
        }
    }

    private static class Symbol {
        int prev;     // prev index of this symbol. -1 for BOS.
        int next;     // next index of tihs symbol. -1 for EOS.
        boolean freeze;  // this symbol is never to be merged.
        String piece;
    }

    private class BytePairEncoder {
        final String normalized;
        final PriorityQueue<SymbolPair> agenda;
        final ArrayList<Symbol> symbols;
        // Reverse merge rules.
        // key: merged symbol, value: pair of original symbols.
        final HashMap<String, Pair<String, String>> revMerge;

        private BytePairEncoder(final String normalized) {
            this.normalized = normalized;
            this.agenda = new PriorityQueue<>();
            this.symbols = new ArrayList<>(normalized.length());
            this.revMerge = new HashMap<>();
        }

        // Create new symbol pair from the symbols from left to right (both inclusive)
        private void maybeAddNewSymbolPair(final int left, final int right) {
            // We do not add anything if we either hit the begin or end of the symbols list
            // or one of the symbols is frozen.
            if (left == -1 || right == -1 ||
                    symbols.get(left).freeze || symbols.get(right).freeze) {
                return;
            }
            final String leftPiece = symbols.get(left).piece;
            final String rightPiece = symbols.get(right).piece;
            //TODO: check if always left = right - 1
            final String piece = leftPiece + rightPiece;
            final Integer id = pieceToIdMap.get(piece);
            if (id == null) { return; }
            final SymbolPair h = new SymbolPair();
            h.left = left;
            h.right = right;
            h.score = getScore(id);
            h.piece = piece;
            agenda.add(h);

            // Makes `rev_merge` for resegmentation.
            if (isUnused(id)) {
                revMerge.put(piece, new Pair<>(leftPiece, rightPiece));
            }
        }
        private void resegment(final String w, final EncodeResult output) {
            final int id = getId(w);
            //TODO: shouldn't -1 be impossible?
            if (id == -1 || !isUnused(id)) {
                output.add(w, id);
                return;
            }
            final Pair<String, String> p = revMerge.get(w);
            if (p == null) {
                // This block will never be called, as `rev_merge` stores all the
                // resegmentation info for unused id.
                output.add(w, id);
                return;
            }
            // Recursively resegment left and right symbols.
            resegment(p.getKey(), output);
            resegment(p.getValue(), output);
        }

        private EncodeResult encode() {
            // Splits the input into codepoints and user defined symbols, e.g. [h,e,l,l,o, ,<msk>]
            int charIndex = 0;
            while (charIndex < normalized.length()) {
                final Symbol s = new Symbol();
                final String prefix = prefixMatcher.findLongestPrefix(normalized, charIndex);
                if (prefix.isEmpty()) {
                    final int cp = normalized.codePointAt(charIndex);
                    s.piece = normalized.substring(charIndex, charIndex + Character.charCount(cp));
                    s.freeze = false;
                } else {
                    s.piece = prefix;
                    s.freeze = true;
                }
                s.prev = charIndex == 0 ? -1 : symbols.size() - 1;
                symbols.add(s);
                charIndex += s.piece.length();
                s.next = charIndex >= normalized.length() ? -1 : symbols.size();
            }

            //if there are no symbols, just return an empty result.
            if (symbols.isEmpty()) { return new EncodeResult(); }

            // Enqueue all symbol pairs
            for (int i = 1; i < symbols.size(); ++i) {
                maybeAddNewSymbolPair(i - 1, i);
            }

            // Main loop, we process symbol pairs, highest scoring ones first
            while (!agenda.isEmpty()) {
                final SymbolPair top = agenda.poll();
                final Symbol topLeft = symbols.get(top.left);
                final Symbol topRight = symbols.get(top.right);

                // `top` is no longer valid (part of it has been overwritten by a previous merge),
                // discard and continue with next candidate.
                if (topLeft.piece.isEmpty() || topRight.piece.isEmpty() ||
                        topLeft.piece.length() + topRight.piece.length() != top.piece.length()) {
                    continue;
                }

                // Merge text into left symbol
                //["a", "left", "right", "b", "c"] -> ["a", "leftright", "", "b", "c"]
                //This does not change the number of symbols but their content
                topLeft.piece = topLeft.piece + topRight.piece;

                // "Blank out" right symbol:
                // Updates prev/next pointers. (we "ignore)
                topLeft.next = topRight.next;
                if (topRight.next >= 0) { //the right symbol is not at a boundary
                    // connect symbol to the right with the current left symbol
                    symbols.get(topRight.next).prev = top.left;
                }
                //delete text from right symbol
                topRight.piece = "";

                // We have new neighbouring pairs, add them as candidates
                maybeAddNewSymbolPair(topLeft.prev, top.left);
                maybeAddNewSymbolPair(top.left, topLeft.next);
            }
            final EncodeResult output = new EncodeResult();
            for (int index = 0; index != -1; index = symbols.get(index).next) {
                //TODO: what do these macros do? Do they just assert the condition?
                //CHECK_GE(index, 0);
                //CHECK_LT(index, static_cast<int>(symbols.size()));
                resegment(symbols.get(index).piece, output);
            }
            return output;
        }
    }
}
