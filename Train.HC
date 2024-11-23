#define mp_cnt 1
#include "Tokenizer";
#ifdef TARGET_X86
asm {  
_LOAD_F32::
  CVTSS2SD XMM0,U32 8[RSP]
  MOVQ RAX,XMM0
  RET1 8
};
_extern _LOAD_F32 F64 F32ToF64(I32 have);
#else
F64 F32ToF64(I32 have) {
  F64 r;
  F64 frac=have&0x7FFffF;
  I64 exp=(have>>23)&0xff;
  I64 sign=have>>31;
  if(!exp&&!frac)
     return 0.;
  frac=1+frac/0x7FFffF;
  r=2.`(exp-I8_MAX)*frac;
  if(sign)
    return -r;
  return r;
} 
#endif
class CGPTConfig {
  I32 max_seq_len;
  I32 vocab_size;
  I32 padded_vocab_size;
  I32 num_layers;
  I32 num_heads;
  I32 channels;
};
class CParamTensors {
  F64* wte; // (V, C)
  F64* wpe; // (maxT, C)
  F64* ln1w; // (L, C)
  F64* ln1b; // (L, C)
  F64* qkvw; // (L, 3*C, C)
  F64* qkvb; // (L, 3*C)
  F64* attprojw; // (L, C, C)
  F64* attprojb; // (L, C)
  F64* ln2w; // (L, C)
  F64* ln2b; // (L, C)
  F64* fcw; // (L, 4*C, C)
  F64* fcb; // (L, 4*C)
  F64* fcprojw; // (L, C, 4*C)
  F64* fcprojb; // (L, C)
  F64* lnfw; // (C)
  F64 * lnfb; // (C)
};
#define NUM_PARAM_TENSORS 16

I64 FillParamSizes(I64 *param_sizes,CGPTConfig *config) {
  I64 Vp = config->padded_vocab_size;
  I64 C = config->channels;
  I64 maxT = config->max_seq_len;
  I64 L = config->num_layers;
  param_sizes[0] = Vp * C; // wte
  param_sizes[1] = maxT * C; // wpe
  param_sizes[2] = L * C; // ln1w
  param_sizes[3] = L * C; // ln1b
  param_sizes[4] = L * (3 * C) * C; // qkvw
  param_sizes[5] = L * (3 * C); // qkvb
  param_sizes[6] = L * C * C; // attprojw
  param_sizes[7] = L * C; // attprojb
  param_sizes[8] = L * C; // ln2w
  param_sizes[9] = L * C; // ln2b
  param_sizes[10] = L * (4 * C) * C; // fcw
  param_sizes[11] = L * (4 * C); // fcb
  param_sizes[12] = L * C * (4 * C); // fcprojw
  param_sizes[13] = L * C; // fcprojb
  param_sizes[14] = C; // lnfw
  param_sizes[15] = C; // lnfb
  I64 i,t=0;
  for(i=0;i!=NUM_PARAM_TENSORS;i++) {
    t+=param_sizes[i];
  }
  return t;
}

CParamTensors *MAllocParamsFromConfig(CGPTConfig *conf) {
  I64 sizes[NUM_PARAM_TENSORS],i;
  FillParamSizes(sizes,conf);
  CParamTensors *ret=CAlloc(sizeof CParamTensors);
  F64 **body=ret;
  for(i=0;i!=NUM_PARAM_TENSORS;i++) {
    body[i]=CAlloc(sizes[i]*sizeof(F64));
  }
  return ret;
}

#define NUM_ACTIVATION_TENSORS 23
class CActivationTensors {
    F64* encoded; // (B, T, C)
    F64* ln1; // (L, B, T, C)
    F64* ln1_mean; // (L, B, T)
    F64* ln1_rstd; // (L, B, T)
    F64* qkv; // (L, B, T, 3*C)
    F64* atty; // (L, B, T, C)
    F64* preatt; // (L, B, NH, T, T)
    F64* att; // (L, B, NH, T, T)
    F64* attproj; // (L, B, T, C)
    F64* residual2; // (L, B, T, C)
    F64* ln2; // (L, B, T, C)
    F64* ln2_mean; // (L, B, T)
    F64* ln2_rstd; // (L, B, T)
    F64* fch; // (L, B, T, 4*C)
    F64* fch_gelu; // (L, B, T, 4*C)
    F64* fcproj; // (L, B, T, C)
    F64* residual3; // (L, B, T, C)
    F64* lnf; // (B, T, C)
    F64* lnf_mean; // (B, T)
    F64* lnf_rstd; // (B, T)
    F64* logits; // (B, T, V)
    F64* probs; // (B, T, V)
    F64* losses; // (B, T)
};

U0 FillActivationSizes(I64 *act_sizes, CGPTConfig *config,I64 B,I64 T) {
    I64 C = config->channels;
    I64 NH = config->num_heads;
    I64 L = config->num_layers;
    I64 Vp = config->padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

CActivationTensors *MAllocActivationFromConfig(CGPTConfig *conf,I64 B,I64 T) {
  I64 sizes[NUM_ACTIVATION_TENSORS],i;
  FillActivationSizes(sizes,conf,B,T);
  CActivationTensors *ret=CAlloc(sizeof CActivationTensors);
  F64 **body=ret;
  for(i=0;i!=NUM_ACTIVATION_TENSORS;i++) {
    body[i]=CAlloc(sizes[i]*sizeof(F64));
  }
  return ret;
}

class CGPT {
  CGPTConfig config;
  CParamTensors *params;
  I64 param_sizes[NUM_PARAM_TENSORS];
  I64 num_params;
  CParamTensors *grads;


  CActivationTensors *acts;
  I64 activation_sizes[NUM_PARAM_TENSORS];
  I64 num_acts;

  CActivationTensors *grad_acts;

  I64 batch_size;
  I64 seq_len;
  I32 *inputs;
  I32 *targets;
  F64 mean_loss;

  F64 *m_memory;
  F64 *v_memory;
}; 

I64 ReadInt(U8 **file) {
  I64 ptr=(*file)(I32*)[0];
  *file+=4;
  return ptr;
}

F64 ReadFloat(U8 **file) {
  F64 ptr=F32ToF64((*file)(I32*)[0]);
  *file+=4;
  return ptr;
}

CGPT *LoadGPTCheckpoint(U8 *fn) {
  CGPT *ret=CAlloc(sizeof CGPT);
  U8 *file=FileRead(fn),*ptr=file;
  I64 magic=ReadInt(&ptr);
  I64 ver=ReadInt(&ptr);
  I64 psizes[NUM_PARAM_TENSORS],i,i2;
  F64 **pmeta,*p;
  /*if(magic!=20240520)
    throw('Magic');
  if(ver!=1) 
    throw('Version');*/

  I64 maxT, V, Vp, L, NH, C;
  maxT=ReadInt(&ptr);
  V=ReadInt(&ptr);
  L=ReadInt(&ptr);
  NH=ReadInt(&ptr);
  C=ReadInt(&ptr);
  Vp=ReadInt(&ptr);


  ret->config.max_seq_len=maxT;
  ret->config.vocab_size=V;
  ret->config.num_layers=L;
  ret->config.num_heads=NH;
  ret->config.channels=C;
  ret->config.padded_vocab_size=Vp;

  ClassRep(&ret->config);

  pmeta=ret->params=MAllocParamsFromConfig(&ret->config);
  ret->num_params=FillParamSizes(psizes,&ret->config);
  ptr=file+256*4;  
  for(i=0;i!=NUM_PARAM_TENSORS;i++) {
    p=pmeta[i];
    for(i2=0;i2!=psizes[i];i2++) {
      p[i2]=ReadFloat(&ptr);
    }
  }


  Free(file);

  ret->mean_loss=-1.;

  return ret;
}


U0 EncoderForward(F64* out,I32* inp, F64* wte, F64* wpe,I64 B,I64 T,I64 C) {
  I64 b,t,i;
// out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
  // inp is (B,T) of I64egers, holding the token ids at each (b,t) position
  // wte is (V,C) of token embeddings, short for "weight token embeddings"
  // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
  for (b=0;b<B;b++) {
    for (t=0;t<T;t++) {
	// seek to the output position in out[b,t,:]
      F64 *out_bt=out+b*T*C+t*C;
// get the index of the token at inp[b, t]
      I64 ix=inp[b*T+t];
// seek to the position in wte corresponding to the token
      F64 *wte_ix=wte+ix*C;
// seek to the position in wpe corresponding to the position
      F64 *wpe_t=wpe+t*C;
// add the two vectors and store the result in out[b,t,:]
      for (i=0;i<C;i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
      }
    }
  }
}

U0 layernorm_forward(F64 *out,F64 *mean,F64 *rstd,
                       F64 *inp,F64 * weight,F64 * bias,
                       I64 B,I64 T,I64 C) {
// reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  // both inp and out are (B,T,C) of the activations
  // mean and rstd are (B,T) buffers, to be used later in backward pass
  // at each position (b,t) of the input, the C-dimensional vector
  // of activations gets normalized, then scaled and shifted
  F64 _eps = 1e-5;
  I64 b,i,i2,t;
  for (b=0;b<B; b++) {
    for (t = 0; t < T; t++) {
// seek to the input position inp[b,t,:]
      F64 *x = inp + b * T * C + t * C;
// calculate the mean
      F64 m = 0.0;
      for (i = 0; i < C; i++) {
        m += x[i];
      }
      m = m/C;
// calculate the variance (without any bias correction)
      F64 v = 0.0;
      for (i2 = 0; i2 < C; i2++) {
        F64 xshift = x[i2] - m;
        v += xshift * xshift;
      }
      v = v/C;
// calculate the rstd (reciprocal standard deviation)
      F64 s = 1.0 / Sqrt(v + _eps);
// seek to the output position in out[b,t,:]
      F64 *out_bt = out + b * T * C + t * C;
      for (i2 = 0; i2 < C; i2++) {
        F64 n = (s * (x[i2] - m)); // normalize
        F64 o = n * weight[i2] + bias[i2]; // scale and shift
        out_bt[i2] = o; // write
      }
// cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}
I64 MPSliceStart(I64 n) {
  return Clamp(n/ToF64(mp_cnt)*Gs->num,0,n);
  
}
I64 MPSliceEnd(I64 n) {
  return Clamp(n/ToF64(mp_cnt)*(Gs->num+1),0,n);
  
}
class CMPSexyMatMul {
  F64 *out;
  F64* inp;
  F64* weight;
  F64* bias;
  I64 B,T,C,OC;
};
U0 MPSexyMatMul(CMPSexyMatMul *struct) {
  I64 B=struct->B;
  I64 T=struct->T;
  I64 C=struct->C;
  I64 OC=struct->OC;
  I64 b,t,o,i,obt;
  F64 val;
  for (obt = MPSliceStart(B*T); obt < MPSliceEnd(B*T); obt++) {
    t=obt%T;
    b=obt/T;
    I64 bt = b * T + t;
    for (o = 0; o < OC; o++) {
      if(struct->bias)
        val=struct->bias[o];
      else
        val=0;
      for (i = 0; i < C; i++) {
        val += struct->inp[bt * C + i] * struct->weight[o*C + i];
      }
      struct->out[bt * OC + o] = val;

    }
  }
}
U0 matmul_forward(F64* out,
                         F64* inp,F64* weight,F64* bias,
                         I64 B, I64 T, I64 C, I64 OC) {
  I64 c;
  CMPSexyMatMul mm;
  mm.out=out;
  mm.inp=inp;
  mm.weight=weight;
  mm.bias=bias;
  mm.B=B;
  mm.T=T;
  mm.C=C;
  mm.OC=OC;
  
  CJob *jobs[mp_cnt];
  for(c=0;c!=mp_cnt;c++) {
    jobs[c]=JobQue(&MPSexyMatMul,&mm,c,0);
  }
  for(c=0;c!=mp_cnt;c++) { 
    JobResGet(jobs[c]);
  }  
}
class CMPAttentionForward {
  F64 *out;
  F64 *preatt;
  F64*att;
  F64*inp;
  I64 B,T,C,NH;
};
U0 MPAttentionForward(CMPAttentionForward *struct) {
  F64 *out=struct->out;
  F64 *preatt=struct->preatt;
  F64*att=struct->att;
  F64*inp=struct->inp;
  I64 B=struct->B,T=struct->T,C=struct->C,NH=struct->NH;

  // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
  // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
  // that holds the pre-attention and post-attention scores (used in backward)
  // output is (B, T, C)
  // attention is the only layer that mixes information across time
  // every other operation is applied at every (b,t) position independently
  // (and of course, no layer mixes information across batch)
  I64 C3 = C*3;
  I64 hs = C / NH; // head size
  F64 scale = 1.0 / Sqrt(hs);

  I64 b,t,h,i,t2,i2,bth,bt;

  for(bth=MPSliceStart(B*T*NH);bth<MPSliceEnd(B*T*NH);bth++) {
    h=bth/(B*T);
    bt=bth%(B*T);
    b=bt/T;
    t=bt%T;
    F64* query_t = inp + b * T * C3 + t * C3 + h * hs;
    F64* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    F64* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // pass 1: calculate query dot key and maxval
    F64 maxval = -10000.0; // TODO something better
    for (t2 = 0; t2 <= t; t2++) {
      F64* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

      // (query_t) dot (key_t2)
      F64 val = 0.0;
      for (i = 0; i < hs; i++) {
        val += query_t[i] * key_t2[i];
      }
      val *= scale;
      if (val > maxval) {
        maxval = val;
      }

      preatt_bth[t2] = val;
    }

    // pass 2: calculate the exp and keep track of sum
    // maxval is being calculated and subtracted only for numerical stability
    F64 expsum = 0.0;
    for (t2 = 0; t2 <= t; t2++) {
      F64 expv = Exp(preatt_bth[t2] - maxval);
      expsum += expv;
      att_bth[t2] = expv;
    }
    F64 expsum_inv;
    if(expsum==0.0)
      expsum_inv=0;
    else
      expsum_inv=1./expsum;

      // pass 3: normalize to get the softmax
    for (t2 = 0; t2 < T; t2++) {
      if (t2 <= t) {
        att_bth[t2] *= expsum_inv;
      } else {
// causal attention mask. not strictly necessary to set to zero here
        // only doing this explicitly for debugging and checking to PyTorch
        att_bth[t2] = 0.0;
      }
    }

    // pass 4: accumulate weighted values I64o the output of attention
    F64* out_bth = out + b * T * C + t * C + h * hs;
    for (i = 0; i < hs; i++) { out_bth[i] = 0.0; }
    for (t2 = 0; t2 <= t; t2++) {
      F64* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
      F64 att_btht2 = att_bth[t2];
      for (i2 = 0; i2 < hs; i2++) {
        out_bth[i2] += att_btht2 * value_t2[i2];
      }
    }
  }
}
U0 attention_forward(F64 *out,
  F64 *preatt,
  F64*att,
  F64*inp,
  I64 B,I64 T,I64 C,I64 NH) {
  I64 done=0,c;
  CMPAttentionForward af;
  af.out=out;
  af.preatt=preatt;
  af.att=att;
  af.inp=inp;
  af.B=B;
  af.T=T;
  af.C=C;
  af.NH=NH;
  
  CJob *jobs[mp_cnt];
  for(c=0;c!=mp_cnt;c++) {
    jobs[c]=JobQue(&MPAttentionForward,&af,c,0);
  }
  for(c=0;c!=mp_cnt;c++) { 
    JobResGet(jobs[c]);
  }  
}


U0 residual_forward(F64* out,F64* inp1,F64* inp2,I64 N) {
  I64 i;
  for (i = 0; i < N; i++) {
    out[i] = inp1[i] + inp2[i];
  }
}
F64 Tanh(F64 x) {
  return (Exp(2*x)-1)/(Exp(2*x)+1);
}
F64 GELU_SCALING_FACTOR=Sqrt(2/pi);
U0 gelu_forward(F64* out,F64* inp,I64 N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    I64 i;
    for (i = 0; i < N; i++) {
        F64 x = inp[i];
        F64 cube = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + Tanh(GELU_SCALING_FACTOR * (x + cube)));
        if(x<-20)
          out[i]=0;
        else if(x>20)
          out[i]=0;

    }
}

U0 crossentropy_forward(F64* losses,
                          F64* probs,I32* targets,
                          I64 B, I64 T,I64 Vp) {
// output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,Vp) of the probabilities
  // input: targets is (B,T) of I64egers giving the correct index in logits
  I64 b,t;
  for (b = 0; b < B; b++) {
    for (t = 0; t < T; t++) {
// loss = -log(probs[target])
      F64* probs_bt = probs + b * T * Vp + t * Vp;
      I64 ix = targets[b * T + t];
      losses[b * T + t] = -Log10(probs_bt[ix])/Log10(exp_1);
    }
  }
}
class CSoftMaxForward {
  F64* probs;
  F64* logits;
  I64 B,T,V,Vp;
};
U0 MPSoftmaxForward(CSoftMaxForward *sm) {
  F64 *probs=sm->probs;
  F64 *logits=sm->logits;
  I64 B=sm->B,T=sm->T,V=sm->V,Vp=sm->Vp;
// output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
  // input: logits is (B,T,Vp) of the unnormalized log probabilities
  // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
  // example: Vp is 50304 and V is 50257
  I64 b,t,i,bt;
  for (bt = MPSliceStart(B*T); bt < MPSliceEnd(B*T); bt++) {
    t=bt%T;
    b=bt/T;
// probs <- softmax(logits)
    F64 * logits_bt = logits + b * T * Vp + t * Vp;
    F64 * probs_bt = probs + b * T * Vp + t * Vp;

    // maxval is only calculated and subtracted for numerical stability
    F64  maxval = -10000.0; // TODO something better
    for (i = 0; i < V; i++) {
      if (logits_bt[i] > maxval) {
        maxval = logits_bt[i];
      }
    }
    F64  sum = 0.0;
    for (i = 0; i < V; i++) {
      probs_bt[i] = Exp(logits_bt[i] - maxval);
      sum += probs_bt[i];
    }
// note we only loop to V, leaving the padded dimensions
    for (i = 0; i < V; i++) {
      probs_bt[i] /= sum;
    }
// for extra super safety we may wish to include this too,
    // forcing the probabilities here to be zero, but it shouldn't matter
    for (i = V; i < Vp; i++) {
      probs_bt[i] = 0.0;
    }
  }
}
U0 softmax_forward(F64 *probs,
  F64 *logits,
  I64 B,I64 T,I64 V,I64 Vp) {
  I64 c;
  CSoftMaxForward sm;
  sm.probs=probs;
  sm.logits=logits;
  sm.B=B;
  sm.T=T;
  sm.V=V;
  sm.Vp=Vp;
  
  CJob *jobs[mp_cnt];
  for(c=0;c!=mp_cnt;c++) {
    jobs[c]=JobQue(&MPSoftmaxForward,&sm,c,0);
  }
  for(c=0;c!=mp_cnt;c++) { 
    JobResGet(jobs[c]);
  }  
}


U0 GPTForward(CGPT *gpt,I32 *inputs,I32 *targets,I64 B,I64 T) {
  I64 V=gpt->config.vocab_size;
  I64 Vp=gpt->config.padded_vocab_size;
  I64 L=gpt->config.num_layers;
  I64 NH=gpt->config.num_heads;
  I64 C=gpt->config.channels;
  CParamTensors *params;
  CActivationTensors *acts;
  I64 l,i;

  if(!gpt->acts) {
    gpt->acts=MAllocActivationFromConfig(&gpt->config,B,T);
    gpt->seq_len=T;
    gpt->batch_size=B;
    gpt->inputs=CAlloc(B*T*sizeof(I32));
    gpt->targets=CAlloc(B*T*sizeof(I32));
  }
  MemCpy(gpt->inputs,inputs,B*T*sizeof(I32));
  if(targets)
    MemCpy(gpt->targets,targets,B*T*sizeof(I32));

  acts=gpt->acts;
  params=gpt->params;
  F64 *residual;
  EncoderForward(acts->encoded,inputs,params->wte,params->wpe,B,T,C);
  for(l=0;l<L;l++) {
    if(l==0)
      residual=acts->encoded;
    else
      residual=acts->residual3+(l-1)*B*T*C;


      // get the poI64ers of the weights for this layer
    F64 *l_ln1w = params->ln1w + l * C;
    F64 *l_ln1b = params->ln1b + l * C;
    F64 *l_qkvw = params->qkvw + l * 3*C * C;
    F64 *l_qkvb = params->qkvb + l * 3*C;
    F64 *l_attprojw = params->attprojw + l * C * C;
    F64 *l_attprojb = params->attprojb + l * C;
    F64 *l_ln2w = params->ln2w + l * C;
    F64 *l_ln2b = params->ln2b + l * C;
    F64 *l_fcw = params->fcw + l * 4*C * C;
    F64 *l_fcb = params->fcb + l * 4*C;
    F64 *l_fcprojw = params->fcprojw + l * C * 4*C;
    F64 *l_fcprojb = params->fcprojb + l * C;

    // get the poI64ers of the activations for this layer
    F64 *l_ln1 = acts->ln1 + l * B * T * C;
    F64 *l_ln1_mean = acts->ln1_mean + l * B * T;
    F64 *l_ln1_rstd = acts->ln1_rstd + l * B * T;
    F64 *l_qkv = acts->qkv + l * B * T * 3*C;
    F64 *l_atty = acts->atty + l * B * T * C;
    F64 *l_preatt = acts->preatt + l * B * NH * T * T;
    F64 *l_att = acts->att + l * B * NH * T * T;
    F64 *l_attproj = acts->attproj + l * B * T * C;
    F64 *l_residual2 = acts->residual2 + l * B * T * C;
    F64 *l_ln2 = acts->ln2 + l * B * T * C;
    F64 *l_ln2_mean = acts->ln2_mean + l * B * T;
    F64 *l_ln2_rstd = acts->ln2_rstd + l * B * T;
    F64 *l_fch = acts->fch + l * B * T * 4*C;
    F64 *l_fch_gelu = acts->fch_gelu + l * B * T * 4*C;
    F64 *l_fcproj = acts->fcproj + l * B * T * C;
    F64 *l_residual3 = acts->residual3 + l * B * T * C;

    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B*T*C);
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
    gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
  }
  residual= acts->residual3 + (L-1) * B * T * C;
  layernorm_forward(acts->lnf, acts->lnf_mean, acts->lnf_rstd, residual, params->lnfw, params->lnfb, B, T, C);
  matmul_forward(acts->logits, acts->lnf, params->wte, NULL, B, T, C, Vp);
  softmax_forward(acts->probs, acts->logits, B, T, V, Vp);

  // also forward the cross-entropy loss function if we have the targets
  if (targets != NULL) {
    crossentropy_forward(gpt->acts->losses, gpt->acts->probs, targets, B, T, Vp);
// for convenience also evaluate the mean loss
    F64 mean_loss = 0.0;
    for (i=0; i<B*T; i++) { mean_loss += gpt->acts->losses[i]; }
    mean_loss /= B*T;
    gpt->mean_loss = mean_loss;
  } else {
// if we don't have targets, we don't have a loss
    gpt->mean_loss = -1.0;
  }
}


I64 sample_mult(F64* probabilities, I64 n, F64 coin) {
// sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  F64 cdf = 0.0;
  I64 i;
  for (i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}



U0 crossentropy_softmax_backward(F64* dlogits,
                           F64* dlosses,F64* probs, I32* targets,
                           I64 B, I64 T,I64 V,I64 Vp) {
    I64 b,t,i;
    // backwards through both softmax and crossentropy
    for (b = 0; b < B; b++) {
        for (t = 0; t < T; t++) {
            F64* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            F64* probs_bt = probs + b * T * Vp + t * Vp;
            F64 dloss = dlosses[b * T + t];
            I64 ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (i = 0; i < V; i++) {
                F64 p = probs_bt[i];
		F64 indicator;
	 	if(i==ix)
		  indicator=1;
		else
		  indicator=0;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

class CMPMatMulBackward {
  F64* dinp,* dweight,* dbias,* dout,* inp,* weight;
  I64 B,T,C,OC;
};
U0 MPMatMulBackward(CMPMatMulBackward *mmb) {
  F64 *dinp=mmb->dinp;
  F64 *dweight=mmb->dweight;
  F64 *dbias=mmb->dbias;
  F64 *dout=mmb->dout;
  F64 *inp=mmb->inp;
  F64 *weight=mmb->weight;
  I64 B=mmb->B;
  I64 T=mmb->T;
  I64 C=mmb->C;
  I64 OC=mmb->OC;

  I64 b,t,o,i,bt;

  // most of the running time is spent here and in matmul_forward
  // this backward could be done in a single "round" of loops
  // but that doesn't afford an efficient parallelization strategy

  // backward I64o inp first, parallelize over B,T
  for(bt=MPSliceStart(B*T);bt<MPSliceEnd(B*T);bt++) {
    b=bt/T;
    t=bt%T;
    F64* dout_bt = dout + b * T * OC + t * OC;
    F64* dinp_bt = dinp + b * T * C + t * C;
    for (o = 0; o < OC; o++) {
      F64* wrow = weight + o*C;
      F64 d = dout_bt[o];
      for (i = 0; i < C; i++) {
        dinp_bt[i] += wrow[i] * d;
      }
    }
  }
// backward into weight/bias, parallelize over output channels OC
  for(o=MPSliceStart(OC);o<MPSliceEnd(OC);o++) {
    for (b = 0; b < B; b++) {
      for (t = 0; t < T; t++) {
        F64* dout_bt2 = dout + b * T * OC + t * OC;
        F64* inp_bt = inp + b * T * C + t * C;
        F64 *dwrow = dweight + o*C;
        d = dout_bt2[o];
        if (dbias != NULL) { dbias[o] += d; }
        for (i = 0; i < C; i++) {
	  dwrow[i] += inp_bt[i] * d;
        }
      }
    }
  }
}

U0 matmul_backward(F64 *dinp,F64 *dweight,F64 *dbias,F64 *dout,F64 *inp,F64 *weight,
  I64 B,I64 T,I64 C,I64 OC) {
  I64 c;
  CMPMatMulBackward mmb;
  mmb.dinp=dinp;
  mmb.dweight=dweight;
  mmb.dbias=dbias;
  mmb.dout=dout;
  mmb.inp=inp;
  mmb.weight=weight;
  mmb.B=B;
  mmb.T=T;
  mmb.C=C;
  mmb.OC=OC;

  CJob *jobs[mp_cnt];
  for(c=0;c!=mp_cnt;c++) {
    jobs[c]=JobQue(&MPMatMulBackward,&mmb,c,0);
  }
  for(c=0;c!=mp_cnt;c++) { 
    JobResGet(jobs[c]);
  }  
}
U0 layernorm_backward(F64* dinp, F64* dweight, F64* dbias,
                        F64* dout, F64* inp, F64* weight, F64* mean, F64* rstd,
                        I64 B, I64 T, I64 C) {
    I64 b,t,i;
    for (b = 0; b < B; b++) {
        for (t = 0; t < T; t++) {
            F64* dout_bt = dout + b * T * C + t * C;
            F64* inp_bt = inp + b * T * C + t * C;
            F64* dinp_bt = dinp + b * T * C + t * C;
            F64 mean_bt = mean[b * T + t];
            F64 rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            F64 dnorm_mean = 0.;
            F64 dnorm_norm_mean = 0.;
            for (i = 0; i < C; i++) {
                F64 norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                F64 dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (i = 0; i < C; i++) {
                norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                F64 dval = 0.;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}
U0 residual_backward(F64*  dinp1,F64* dinp2,F64 * dout,I64 N) {
  I64 i;
  for (i = 0; i < N; i++) {
    dinp1[i] += dout[i];
    dinp2[i] += dout[i];
  }
}

U0 gelu_backward(F64* dinp, F64* inp, F64* dout, I64 N) {
  I64 i;
  for (i = 0; i < N; i++) {
    F64 x = inp[i];
    if(-20.<=x<=20.) {
      F64 cube = 0.044715 * x * x * x;
      F64 tanh_arg = GELU_SCALING_FACTOR * (x + cube);
      F64 tanh_out = Tanh(tanh_arg);
      F64 coshf_out = Cosh(tanh_arg);
      F64 sech_out = 1.0 / (coshf_out * coshf_out);
      F64 local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
      dinp[i] += local_grad * dout[i];
    } else if(x>20)
      dinp[i]=x;
    else if(x<-20)
      x=-1;
  }
}


U0 attention_backward(F64* dinp, F64* dpreatt, F64* datt,
                        F64* dout, F64* inp, F64* att,
                        I64 B,I64 T,I64 C,I64 NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    I64 C3 = C*3;
    I64 hs = C / NH; // head size
    F64 scale = 1. / Sqrt(hs);
    I64 b,t,h,t2,t3,i;

    for (b = 0; b < B; b++) {
        for (t = 0; t < T; t++) {
            for (h = 0; h < NH; h++) {
                F64* att_bth = att + b*NH*T*T + h*T*T + t*T;
                F64* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                F64* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                F64* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                F64* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                F64* dout_bth = dout + b * T * C + t * C + h * hs;
                for (t2 = 0; t2 <= t; t2++) {
                    F64* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    F64* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (t2 = 0; t2 <= t; t2++) {
                    for (t3 = 0; t3 <= t; t3++) {
                        F64 indicator;
			if(t2 == t3)
			  indicator=1.0;
			else
			  indicator=0.0;
                        F64 local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (t2 = 0; t2 <= t; t2++) {
                    F64* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    F64* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}
U0 encoder_backward(F64* dwte, F64* dwpe,
                      F64* dout, I32* inp,
                      I64 B, I64 T, I64 C) {
    I64 b,t,i;
    for (b = 0; b < B; b++) {
        for (t = 0; t < T; t++) {
            F64* dout_bt = dout + b * T * C + t * C;
            I64 ix = inp[b * T + t];
            F64* dwte_ix = dwte + ix * C;
            F64* dwpe_t = dwpe + t * C;
            for (i = 0; i < C; i++) {
                F64 d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

U0 GPTZeroGrad(CGPT *g) {
  F64 **t;
  I64 i;
  if(g->grads) {
    t=g->grads;
    for(i=0;i!=NUM_PARAM_TENSORS;i++)
      MemSet(t[i],0,MSize(t[i]));
  }
  if(g->grad_acts) {
    t=g->grad_acts;
    for(i=0;i!=NUM_ACTIVATION_TENSORS;i++)
      MemSet(t[i],0,MSize(t[i]));
  }
}
U0 GPTBackward(CGPT *g) {
  if(!g->grad_acts) {
    g->grad_acts=MAllocActivationFromConfig(&g->config,g->batch_size,g->seq_len);
    g->grads=MAllocParamsFromConfig(&g->config);
    GPTZeroGrad(g);
  }
  I64 B = g->batch_size;
  I64 T = g->seq_len;
  I64 V = g->config.vocab_size;
  I64 Vp = g->config.padded_vocab_size;
  I64 L = g->config.num_layers;
  I64 NH = g->config.num_heads;
  I64 C = g->config.channels;
  I64 i,l;
  CParamTensors *params = g->params; // for brevity
  CParamTensors *grads = g->grads;
  CActivationTensors *acts = g->acts;
  CActivationTensors *grads_acts = g->grad_acts;


  // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
  // technically this is a small, inline backward() pass of calculating
  // total, final loss as the mean over all losses over all (B,T) positions in the batch
  F64 dloss_mean = 1.0 / (B*T);
  for (i = 0; i < B*T; i++) { grads_acts->losses[i] = dloss_mean; }

  crossentropy_softmax_backward(grads_acts->logits, grads_acts->losses, acts->probs, g->targets, B, T, V, Vp);
  matmul_backward(grads_acts->lnf, grads->wte, NULL, grads_acts->logits, acts->lnf, params->wte, B, T, C, Vp);
  F64* residual = acts->residual3 + (L-1) * B * T * C; // last layer's residual
  F64* dresidual = grads_acts->residual3 + (L-1) * B * T * C; // write to last layer's residual
  layernorm_backward(dresidual, grads->lnfw, grads->lnfb, grads_acts->lnf, residual, params->lnfw, acts->lnf_mean, acts->lnf_rstd, B, T, C);

  for (l = L-1; l >= 0; l--) {

    if(!l)
      residual= acts->encoded;
    else
      residual=acts->residual3 + (l-1) * B * T * C;
    if(!l)
      dresidual= grads_acts->encoded;
    else
      dresidual=grads_acts->residual3 + (l+-1) * B * T * C;

      // get the pointers of the weights for this layer
    F64* l_ln1w = params->ln1w + l * C;
    F64* l_qkvw = params->qkvw + l * 3*C * C;
    F64* l_attprojw = params->attprojw + l * C * C;
    F64* l_ln2w = params->ln2w + l * C;
    F64* l_fcw = params->fcw + l * 4*C * C;
    F64* l_fcprojw = params->fcprojw + l * C * 4*C;
// get the pointers of the gradients of the weights for this layer
    F64* dl_ln1w = grads->ln1w + l * C;
    F64* dl_ln1b = grads->ln1b + l * C;
    F64* dl_qkvw = grads->qkvw + l * 3*C * C;
    F64* dl_qkvb = grads->qkvb + l * 3*C;
    F64* dl_attprojw = grads->attprojw + l * C * C;
    F64* dl_attprojb = grads->attprojb + l * C;
    F64* dl_ln2w = grads->ln2w + l * C;
    F64* dl_ln2b = grads->ln2b + l * C;
    F64* dl_fcw = grads->fcw + l * 4*C * C;
    F64* dl_fcb = grads->fcb + l * 4*C;
    F64* dl_fcprojw = grads->fcprojw + l * C * 4*C;
    F64* dl_fcprojb = grads->fcprojb + l * C;
// get the pointers of the activations for this layer
    F64* l_ln1 = acts->ln1 + l * B * T * C;
    F64* l_ln1_mean = acts->ln1_mean + l * B * T;
    F64* l_ln1_rstd = acts->ln1_rstd + l * B * T;
    F64* l_qkv = acts->qkv + l * B * T * 3*C;
    F64* l_atty = acts->atty + l * B * T * C;
    F64* l_att = acts->att + l * B * NH * T * T;
    F64* l_residual2 = acts->residual2 + l * B * T * C;
    F64* l_ln2 = acts->ln2 + l * B * T * C;
    F64* l_ln2_mean = acts->ln2_mean + l * B * T;
    F64* l_ln2_rstd = acts->ln2_rstd + l * B * T;
    F64* l_fch = acts->fch + l * B * T * 4*C;
    F64* l_fch_gelu = acts->fch_gelu + l * B * T * 4*C;
// get the pointers of the gradients of the activations for this layer
    F64* dl_ln1 = grads_acts->ln1 + l * B * T * C;
    F64* dl_qkv = grads_acts->qkv + l * B * T * 3*C;
    F64* dl_atty = grads_acts->atty + l * B * T * C;
    F64* dl_preatt = grads_acts->preatt + l * B * NH * T * T;
    F64* dl_att = grads_acts->att + l * B * NH * T * T;
    F64* dl_attproj = grads_acts->attproj + l * B * T * C;
    F64* dl_residual2 = grads_acts->residual2 + l * B * T * C;
    F64* dl_ln2 = grads_acts->ln2 + l * B * T * C;
    F64* dl_fch = grads_acts->fch + l * B * T * 4*C;
    F64* dl_fch_gelu = grads_acts->fch_gelu + l * B * T * 4*C;
    F64* dl_fcproj = grads_acts->fcproj + l * B * T * C;
    F64* dl_residual3 = grads_acts->residual3 + l * B * T * C;

    // backprop this layer
    residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
    matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
    gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
    matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
    layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
    residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
    matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
    attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
    matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
  }
  encoder_backward(grads->wte, grads->wpe, grads_acts->encoded, g->inputs, B, T, C);
}

U0 GPTUpdate(CGPT *model,F64 learning_rate, F64 beta1, F64 beta2, F64 _eps, F64 weight_decay, I64 t) {
// reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  // lazily allocate the memory for m_memory and v_memory
  if (model->m_memory == NULL) {
    model->m_memory = CAlloc(model->num_params*sizeof(F64));
    model->v_memory = CAlloc(model->num_params*sizeof(F64));
  }

  F64 **pmem=model->params,**gmem=model->grads;

  I64 i,param_sizes[NUM_PARAM_TENSORS],p;
 FillParamSizes(param_sizes,&model->config);
  for(p=0;p!=NUM_PARAM_TENSORS;p++)
    for (i = 0; i < param_sizes[p]; i++) {
      F64 param = pmem[p][i];
      F64 grad = gmem[p][i];

 	     // update the first moment (momentum)
      F64 m = beta1 * model->m_memory[i] + (1. - beta1) * grad;
// update the second moment (RMSprop)
      F64 v = beta2 * model->v_memory[i] + (1. - beta2) * grad * grad;
// bias-correct both moments
      F64 m_hat = m / (1. - beta1`t);
      F64 v_hat = v / (1. - beta1`t);

      // update
      model->m_memory[i] = m;
      model->v_memory[i] = v;
      pmem[p][i] -= learning_rate * (m_hat / (Sqrt(v_hat) + _eps) + weight_decay * param);
    }
}

//Retursn static buffer
U8 *SafeDollar(U8 *s) {
  if(!s) return "";
  static U8 buf[STR_LEN];
  I64 l=StrLen(s),c,p=0;
  for(c=0;c!=l;c++) {
    if(s[c]!='\d')
      buf[p++]=s[c];
    else {
     buf[p++]='\d';
     buf[p++]='\d';
    }
  }
  buf[p]=0;
  return buf; 
}

U0 GPTRun(CGPT *g,U8 *string) {
  if(!g->batch_size||!g->seq_len) {
    PrintErr("Train me first.");
    throw('Train');
  }
  I32 *toks=TokenizeText(tokenizer,string);
  "PROMPT(%s)>>>",string;
  I64 B=g->batch_size,T=g->seq_len,genT=32;
  I32 *inp=CAlloc(4*B*T);
  I32 *gen_tokens=CAlloc(4*B*T);
  I64 t,next_token;
  MemSetU32(inp,tokenizer->end_token,B*T);
  MemSetU32(gen_tokens,tokenizer->end_token,B*T);
  for(t=1;toks[t-1]!=tokenizer->end_token;t++) {
    inp[t]=toks[t-1];
  }
  for(t;t<genT;t++) {
    GPTForward(g, inp, NULL, B, T);
    F64 *probs=g->acts->probs + (t-1) * g->config.padded_vocab_size;
    next_token=sample_mult(probs,g->config.padded_vocab_size,.1);
    "%s",SafeDollar(tokenizer->tokens[next_token]);
    if(next_token==tokenizer->end_token)
	break;
    inp[t]=next_token;
  }
  "\n";
  Free(inp);
  Free(gen_tokens);
  Free(toks);
}
U0 GPTTrain(CGPT *g,I64 steps,...) {
  I32 **dataset=CAlloc(sizeof(I32*)*argc),*tokens;
  I32 **targets=CAlloc(sizeof(I32*)*argc);
  I64 B=argc,T=0,t,run_cnt,s,tc;
  for(t=0;t!=argc;t++) {
    tokens=dataset[t]=TokenizeText(tokenizer,argv[t]);
    for(tc=0;tokens[tc]!=tokenizer->end_token;tc++)
      ;
    T=MaxI64(MinI64(tc*2+1,1024),T);
    targets[t]=CAlloc((tc+2)*sizeof(I32));
    MemCpy(targets[t],tokens+1,sizeof(I32)*(tc-1));
    targets[t][tc-1]=tokenizer->end_token;
  }
  I32 *inp=CAlloc(4*B*T);
  I32 *shifted=CAlloc(4*B*T);
  MemSetU32(inp,tokenizer->end_token,B*T);
  MemSetU32(shifted,tokenizer->end_token,B*T);
  for(t=0;t!=argc;t++) {
    tokens=dataset[t];
"INPUT:%d>>>",t;
    for(tc=0;tokens[tc]!=tokenizer->end_token;tc++) {
	"%s",tokenizer->tokens[tokens[tc]];
      inp[t*T+tc]=tokens[tc];
    }
"\n";
  }
  for(t=0;t!=argc;t++) {
    tokens=targets[t];
"TARGET:%d>>>",t;
    for(tc=0;tokens[tc]!=tokenizer->end_token;tc++) {
	"%s",tokenizer->tokens[tokens[tc]];
      shifted[t*T+tc]=tokens[tc];
    }
    "\n";
  }
  for(s=0;s!=steps;s++) {
    GPTForward(g,inp,shifted,B,T);
    GPTZeroGrad(g);
    GPTBackward(g);
    GPTUpdate(g,1e-4,0.9,0.999,1e-8,0.0,s+1);
"STEP %d,COMPLETE\n",s;
//    GPTRun(g,"I eat");
  }
  for(t=0;t!=argc;t++) {
    Free(dataset[t]);
    Free(targets[t]);
  }
  Free(inp);
  Free(shifted);
  Free(targets);
  Free(dataset);
}

CGPT *gpt=LoadGPTCheckpoint("gpt2_124M.bin");
GPTTrain(gpt,6,"I eat carrots");
GPTRun(gpt,"I eat");
