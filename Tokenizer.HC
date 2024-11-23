#exe {Cd(__DIR__);;};;
class CTrie {
  I64 token;
  CTrie *children[0x100];
};
class CTokenizer {
  I64 size;
  U8 **tokens;
  I64 end_token;
  CTrie trie;
};
U0 AddToken(CTrie *t,I64 token,U8 *name) {
  if(!*name) {
    t->token=token;
    return;
  }
  if(!t->children[*name]) {
    t->children[*name]=CAlloc(sizeof CTrie);
  }
  AddToken(t->children[*name],token,name+1);
}
CTokenizer *TokenizerInit(U8 *file) {
  CTokenizer *tok=CAlloc(sizeof CTokenizer);
  I32 *header;
  U8 *body;
  I32 version;
  I32 eot,size,i,size2;
  file=FileRead(file);
  if(!file)
    throw('Token');
  header=file;
  if(header[0]!=20240328)
    throw('Token');
  version=header[1];
  size=tok->size=header[2];
  switch(version) {
    case 1:
      tok->end_token=50256;
      break;
    case 2:
      tok->end_token=header[3];
      break;
    default:
      throw('Token');
      break;
  }
  tok->tokens=CAlloc(8*size);
  body=file+256*4;
  tok->trie.token=tok->end_token;
  for(i=0;i!=size;i++) {
    size2=*(body++);
    tok->tokens[i]=CAlloc(size2+1);
    MemCpy(tok->tokens[i],body,size2);
    AddToken(&tok->trie,i,tok->tokens[i]);
    body+=size2;
  }
  Free(file);
  return tok;
}
I32 TokenizeText0(CTokenizer *t,U8 **_text) {
  CTrie *trie=&t->trie;
  I32 best=t->end_token;
  U8 *text=*_text;
loop:;
  if(!*text)  {
    *_text=text;
    return best;
  }
  if(trie->children[*text]) {
    trie=trie->children[*text];
    text++;
    if(trie->token) {
      *_text=text;
      best=trie->token;
    }
    goto loop;
  }
  return best;  
} 
I32 *TokenizeText(CTokenizer *t,U8 *text) {
  U8 *otext=MStrUtil(text,SUF_REM_TRAILING|SUF_REM_LEADING);
  text=otext;
  I32 *ret=CAlloc(4*StrLen(text)+4);
  I64 ptr=0;
  while(*text) {
    ret[ptr++]=TokenizeText0(t,&text);
  }
  ret[ptr]=t->end_token;
  Free(otext);
  return ret;
}

CTokenizer *tokenizer=TokenizerInit("gpt2_tokenizer.bin");
#if __CMD_LINE__
I32 *toks=TokenizeText(tokenizer,"Aiwnios issa great compiler");
I64 idx=0;
for(idx=0;toks[idx]!=tokenizer->end_token;idx++)
  "%s",tokenizer->tokens[toks[idx]];
"\n";
#endif