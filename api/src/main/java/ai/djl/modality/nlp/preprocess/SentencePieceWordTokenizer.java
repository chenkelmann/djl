package ai.djl.modality.nlp.preprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Tokenizes using a replacement character for space. This is necessary for using SentencePiece Preprocessing.
 * This is the analog to the C++ method {@code SplitIntoWords} in the original SentencePiece implementation:
 * https://github.com/google/sentencepiece/blob/master/src/model_interface.cc
 *
 * This is not the actual sentence piece tokenization but the preprocessing step that turns texts into words that
 * in turn get tokenized using sentence pieces.
 */
public class SentencePieceWordTokenizer implements Tokenizer {

    /**
     * The unicode codepoint to use to symbolize a space. Sentence piece does not use the normal space character for
     * word boundaries to allow tokens like "New York" to be treated as single words.
     */
    public static final int SPACE_SYMBOL_CP = 0x2581;

    /**
     * A String containing only the codepoint {@link SentencePieceWordTokenizer#SPACE_SYMBOL}
     */
    public static final String SPACE_SYMBOL = new StringBuilder(1).appendCodePoint(SPACE_SYMBOL_CP).toString();

    private final boolean treatWhitespaceAsSuffix;

    /**
     * Builds a Tokenizer that splits sentences into words for further sentence piece tokenization.
     * @param treatWhitespaceAsSuffix true: the {@link SentencePieceWordTokenizer#SPACE_SYMBOL} is left at the end
     *                                of the previous token, false: it is left at the beginning of the next token.
     */
    public SentencePieceWordTokenizer(final boolean treatWhitespaceAsSuffix) {
        this.treatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
    }


    @Override
    public List<String> tokenize(final String sentence) {
        final StringBuilder currentWord = new StringBuilder();
        final ArrayList<String> result = new ArrayList<>();
        sentence.codePoints().forEach((cp) -> {
            final boolean isWs = cp == SPACE_SYMBOL_CP;
            // We have reached a word boundary, add new word
            if (isWs) {
                // We want the WS as suffix, so add it to the current word
                // before adding the new word to the result
                if (treatWhitespaceAsSuffix) {
                    currentWord.appendCodePoint(cp);
                }
                // Add word if we have accumulated anything
                if (currentWord.length() > 0) {
                    result.add(currentWord.toString());
                    currentWord.setLength(0);
                }
                // If we want WS as prefix, add it here
                if (!treatWhitespaceAsSuffix) {
                    currentWord.appendCodePoint(cp);
                }
            } else {
                currentWord.appendCodePoint(cp);
            }
        });
        // Do not forget to add last word
        if  (currentWord.length() > 0) {
            result.add(currentWord.toString());
        }
        // Done!
        return result;
    }

    @Override
    public String buildSentence(final List<String> tokens) {
        return tokens.stream().collect(Collectors.joining());
    }
}
