#exe {Cd(__DIR__);;};;

#include "Tokenizer.HC";
#include "Numbers-main/Matrix.HC";


//Prepare dataset



//Tokenizer Input

CTokenizer *t=TokenizerInit("gpt2_tokenizer.bin");
I32 *toks=TokenizeText(t,"Aiwnios is a great compiler"); //<<--Change this
I64 idx=0;
//for(idx=0;toks[idx]!=t->end_token;idx++)
//  "%s",t->tokens[toks[idx]];
//"\n";



//Encoding
//embedding matrix is a 2D vector 
//(shape [V, D]), where each row corresponds to a token's embedding.
for(idx=0;toks[idx]!=t->end_token;idx++)
  CreateMatrix( t->tokens[toks[idx]];
"\n";

