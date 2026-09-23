#define _POSIX_C_SOURCE 200809L
#include "model.h"
#include "data.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s predict <model.sjev> --context \"text\" --option \"a\" --option \"b\" ...\n"
        "  %s train TRAIN.jsonl --validation VALID.jsonl --output MODEL.sjev [options]\n"
        "  %s eval MODEL.sjev TEST.jsonl\n"
        "  %s grad-check MODEL.sjev DATA.jsonl\n"
        "  %s predict --help | train --help | eval --help\n"
        "\n"
        "Train options (defaults interoperable with jevlike.train):\n"
        "  --width INT            default 64\n"
        "  --rank INT             default 64\n"
        "  --context-tokens INT   default 192\n"
        "  --option-tokens INT    default 32\n"
        "  --epochs INT           default 8\n"
        "  --batch-size INT       default 64\n"
        "  --learning-rate FLOAT  default 0.002\n"
        "  --weight-decay FLOAT   default 1e-4\n"
        "  --seed INT             default 7\n"
        "  --init-model PATH      fine-tune from existing .sjev (fresh Adam)\n"
        "  --output PATH          (train)\n"
        "  --validation PATH      (train)\n",
        prog, prog, prog, prog, prog);
}

static void json_escape(FILE *out, const char *s) {
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        if (c == '"') { fputs("\\\"", out); }
        else if (c == '\\') { fputs("\\\\", out); }
        else if (c == '\n') { fputs("\\n", out); }
        else if (c == '\r') { fputs("\\r", out); }
        else if (c == '\t') { fputs("\\t", out); }
        else if (c < 0x20) { fprintf(out, "\\u%04x", c); }
        else { fputc(c, out); }
    }
    fputc('"', out);
}

static int cmd_predict(int argc, char **argv) {
    const char *model_path = NULL;
    const char *context = NULL;
    char **options = NULL;
    int n_options=0, cap=0;
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"--help")==0||strcmp(argv[i],"-h")==0){ print_usage(argv[0]); free(options); return 0; }
        else if(strcmp(argv[i],"--context")==0){
            if(i+1>=argc){ fprintf(stderr,"--context requires argument\n"); free(options); return 1; }
            context=argv[++i];
        } else if(strcmp(argv[i],"--option")==0){
            if(i+1>=argc){ fprintf(stderr,"--option requires argument\n"); free(options); return 1; }
            const char *val=argv[++i];
            if(n_options>=cap){ int ncap=cap?cap*2:4; char **tmp=(char**)realloc(options,(size_t)ncap*sizeof(char*)); if(!tmp){fprintf(stderr,"oom\n"); free(options); return 1;} options=tmp; cap=ncap; }
            options[n_options++]=(char*)val;
        } else if(strncmp(argv[i],"--",2)==0){ fprintf(stderr,"unknown flag %s\n",argv[i]); free(options); return 1; }
        else {
            if(!model_path) model_path=argv[i];
            else { fprintf(stderr,"unexpected positional %s\n",argv[i]); free(options); return 1; }
        }
    }
    if(!model_path){ fprintf(stderr,"missing <model.sjev>\n"); free(options); return 1; }
    if(!context){ fprintf(stderr,"missing --context\n"); free(options); return 1; }
    if(n_options<2){ fprintf(stderr,"need >=2 --option\n"); free(options); return 1; }
    struct sjev_model *m=NULL;
    char err[512]={0};
    if(sjev_model_load(model_path,&m,err,sizeof(err))!=0){ fprintf(stderr,"load %s: %s\n",model_path,err); free(options); return 1; }
    float *logits=(float*)malloc((size_t)n_options*sizeof(float));
    float *probs=(float*)malloc((size_t)n_options*sizeof(float));
    if(!logits||!probs){ fprintf(stderr,"oom\n"); sjev_model_free(m); free(logits); free(probs); free(options); return 1; }
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    if(sjev_predict(m,context,(const char*const*)options,n_options,logits,probs,err,sizeof(err))!=0){ fprintf(stderr,"predict: %s\n",err); sjev_model_free(m); free(logits); free(probs); free(options); return 1; }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    printf("[\n");
    for(int i=0;i<n_options;i++){
        printf("  {\n    \"option\": "); json_escape(stdout, options[i]); printf(",\n");
        printf("    \"probability\": %.10f,\n", (double)probs[i]);
        printf("    \"logit\": %.10f\n", (double)logits[i]);
        printf("  }%s\n", i+1<n_options?",":"");
    }
    printf("]\n");
    fprintf(stderr,"inference_time_ms: %.3f\n", ms);
    sjev_model_free(m); free(logits); free(probs); free(options);
    return 0;
}

static int cmd_train(int argc, char **argv) {
    const char *train_path=NULL;
    const char *valid_path=NULL;
    const char *output_path="runs/model.sjev";
    int width=64, rank=64, ctx_tok=192, opt_tok=32;
    int epochs=8, batch_size=64;
    double lr=0.002, wd=1e-4;
    int64_t seed=7;
    const char *init_model_path=NULL;
    int have_width=0, have_rank=0, have_ctx=0, have_opt=0;
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"--help")==0||strcmp(argv[i],"-h")==0){ print_usage(argv[0]); return 0; }
        else if(strcmp(argv[i],"--validation")==0){ if(i+1>=argc){fprintf(stderr,"--validation requires arg\n");return 1;} valid_path=argv[++i]; }
        else if(strcmp(argv[i],"--output")==0){ if(i+1>=argc){fprintf(stderr,"--output requires arg\n");return 1;} output_path=argv[++i]; }
        else if(strcmp(argv[i],"--width")==0){ if(i+1>=argc){fprintf(stderr,"--width requires arg\n");return 1;} width=atoi(argv[++i]); have_width=1; }
        else if(strcmp(argv[i],"--rank")==0){ if(i+1>=argc){fprintf(stderr,"--rank requires arg\n");return 1;} rank=atoi(argv[++i]); have_rank=1; }
        else if(strcmp(argv[i],"--context-tokens")==0){ if(i+1>=argc){fprintf(stderr,"--context-tokens requires arg\n");return 1;} ctx_tok=atoi(argv[++i]); have_ctx=1; }
        else if(strcmp(argv[i],"--option-tokens")==0){ if(i+1>=argc){fprintf(stderr,"--option-tokens requires arg\n");return 1;} opt_tok=atoi(argv[++i]); have_opt=1; }
        else if(strcmp(argv[i],"--epochs")==0){ if(i+1>=argc){fprintf(stderr,"--epochs requires arg\n");return 1;} epochs=atoi(argv[++i]); }
        else if(strcmp(argv[i],"--batch-size")==0){ if(i+1>=argc){fprintf(stderr,"--batch-size requires arg\n");return 1;} batch_size=atoi(argv[++i]); }
        else if(strcmp(argv[i],"--learning-rate")==0){ if(i+1>=argc){fprintf(stderr,"--learning-rate requires arg\n");return 1;} lr=atof(argv[++i]); }
        else if(strcmp(argv[i],"--weight-decay")==0){ if(i+1>=argc){fprintf(stderr,"--weight-decay requires arg\n");return 1;} wd=atof(argv[++i]); }
        else if(strcmp(argv[i],"--seed")==0){ if(i+1>=argc){fprintf(stderr,"--seed requires arg\n");return 1;} seed=atoll(argv[++i]); }
        else if(strcmp(argv[i],"--init-model")==0){ if(i+1>=argc){fprintf(stderr,"--init-model requires arg\n");return 1;} init_model_path=argv[++i]; }
        else if(strncmp(argv[i],"--",2)==0){ fprintf(stderr,"unknown flag %s\n",argv[i]); return 1; }
        else {
            if(!train_path) train_path=argv[i];
            else { fprintf(stderr,"unexpected positional %s\n",argv[i]); return 1; }
        }
    }
    if(!train_path){ fprintf(stderr,"missing TRAIN.jsonl\n"); return 1; }
    if(!valid_path){ fprintf(stderr,"missing --validation\n"); return 1; }
    char err[1024]={0};
    struct dataset train, valid;
    if(dataset_load(train_path,&train,err,sizeof(err))!=0){ fprintf(stderr,"train load: %s\n",err); return 1; }
    if(dataset_load(valid_path,&valid,err,sizeof(err))!=0){ fprintf(stderr,"valid load: %s\n",err); dataset_free(&train); return 1; }
    struct sjev_model *m=NULL;
    if(init_model_path){
        if(sjev_model_load(init_model_path,&m,err,sizeof(err))!=0){ fprintf(stderr,"init-model load %s: %s\n",init_model_path,err); dataset_free(&train); dataset_free(&valid); return 1; }
        if((have_width && (int)m->width!=(int)width) || (have_rank && (int)m->rank!=(int)rank) ||
           (have_ctx && (int)m->context_tokens!=(int)ctx_tok) || (have_opt && (int)m->option_tokens!=(int)opt_tok)){
            fprintf(stderr,"--init-model dims (W=%u R=%u C=%u O=%u) conflict with explicit flags\n",
                m->width, m->rank, m->context_tokens, m->option_tokens);
            sjev_model_free(m); dataset_free(&train); dataset_free(&valid); return 1;
        }
        fprintf(stderr,"fine-tuning from %s (W=%u R=%u C=%u O=%u), fresh Adam state\n",
            init_model_path, m->width, m->rank, m->context_tokens, m->option_tokens);
    } else {
        if(sjev_model_init(&m, (uint32_t)width,(uint32_t)rank,(uint32_t)ctx_tok,(uint32_t)opt_tok,(uint64_t)seed,err,sizeof(err))!=0){ fprintf(stderr,"init: %s\n",err); dataset_free(&train); dataset_free(&valid); return 1; }
    }
    struct adam_state adam;
    if(adam_alloc(m,&adam,(float)lr,0.9f,0.999f,1e-8f,(float)wd)!=0){ fprintf(stderr,"adam alloc oom\n"); sjev_model_free(m); dataset_free(&train); dataset_free(&valid); return 1; }
    int rc = train_loop(m, &train, &valid, epochs, batch_size, &adam, output_path, err, sizeof(err));
    if(rc!=0){ fprintf(stderr,"train_loop: %s\n",err); }
    adam_free(m,&adam);
    sjev_model_free(m);
    dataset_free(&train); dataset_free(&valid);
    return rc;
}

static int cmd_eval(int argc, char **argv) {
    if(argc<2){ fprintf(stderr,"eval requires MODEL.sjev TEST.jsonl\n"); return 1; }
    const char *model_path=NULL,*test_path=NULL;
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"--help")==0||strcmp(argv[i],"-h")==0){ print_usage(argv[0]); return 0; }
        else if(strncmp(argv[i],"--",2)==0){ fprintf(stderr,"unknown flag %s\n",argv[i]); return 1; }
        else {
            if(!model_path) model_path=argv[i];
            else if(!test_path) test_path=argv[i];
            else { fprintf(stderr,"unexpected %s\n",argv[i]); return 1; }
        }
    }
    if(!model_path||!test_path){ fprintf(stderr,"eval MODEL.sjev TEST.jsonl\n"); return 1; }
    char err[1024]={0};
    struct sjev_model *m=NULL;
    if(sjev_model_load(model_path,&m,err,sizeof(err))!=0){ fprintf(stderr,"load %s: %s\n",model_path,err); return 1; }
    struct dataset ds;
    if(dataset_load(test_path,&ds,err,sizeof(err))!=0){ fprintf(stderr,"load %s: %s\n",test_path,err); sjev_model_free(m); return 1; }
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    float nll, top1, top3;
    if(eval_dataset(m,&ds,&nll,&top1,&top3,err,sizeof(err))!=0){ fprintf(stderr,"eval: %s\n",err); dataset_free(&ds); sjev_model_free(m); return 1; }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    double ece=0;
    {
        size_t B=ds.n;
        float *conf=(float*)malloc(B*sizeof(float));
        int *pred=(int*)malloc(B*sizeof(int));
        int *labels=(int*)malloc(B*sizeof(int));
        if(conf&&pred&&labels){
            for(size_t i=0;i<ds.n;i++){
                struct forward_cache c;
                if(forward_one(m,&ds.examples[i],&c,err,sizeof(err))!=0) continue;
                float best=-1e30; int besti=-1;
                for(int k=0;k<c.N;k++) if(c.logits[k]>best){best=c.logits[k]; besti=k;}
                float maxp=-1; for(int k=0;k<c.N;k++) if(c.probs[k]>maxp) maxp=c.probs[k];
                conf[i]=maxp; pred[i]=besti; labels[i]=c.label;
                cache_free(&c);
            }
            for(int bin=0;bin<10;bin++){
                double lo=bin*0.1, hi=lo+0.1;
                int cnt=0, correct=0; double sum_conf=0;
                for(size_t i=0;i<ds.n;i++) if(conf[i]>=lo && conf[i]<hi+ (bin==9?0.1:0)) {cnt++; sum_conf+=conf[i]; if(pred[i]==labels[i]) correct++;}
                if(cnt>0){
                    double acc=(double)correct/cnt;
                    double avg_conf=sum_conf/cnt;
                    ece += (double)cnt/ds.n * fabs(acc - avg_conf);
                }
            }
        }
        free(conf); free(pred); free(labels);
    }
    printf("{\"test_nll\": %.6f, \"top1\": %.6f, \"top3\": %.6f, \"ece\": %.6f, \"examples\": %zu, \"inference_ms\": %.2f}\n", nll, top1, top3, ece, ds.n, ms);
    dataset_free(&ds);
    sjev_model_free(m);
    return 0;
}

static int cmd_grad_check(int argc, char **argv) {
    if(argc<2){ fprintf(stderr,"grad-check MODEL.sjev DATA.jsonl\n"); return 1; }
    const char *model_path=argv[0];
    const char *data_path=argv[1];
    char err[1024]={0};
    struct sjev_model *m=NULL;
    if(sjev_model_load(model_path,&m,err,sizeof(err))!=0){ fprintf(stderr,"load %s: %s\n",model_path,err); return 1; }
    struct dataset ds;
    if(dataset_load(data_path,&ds,err,sizeof(err))!=0){ fprintf(stderr,"load %s: %s\n",data_path,err); sjev_model_free(m); return 1; }
    struct sjev_grad grad;
    if(grad_alloc(m,&grad,err,sizeof(err))!=0){ fprintf(stderr,"grad alloc %s\n",err); dataset_free(&ds); sjev_model_free(m); return 1; }
    grad_zero(m,&grad);
    double loss_sum=0;
    printf("{\"logits\": [");
    for(size_t i=0;i<ds.n;i++){
        struct forward_cache c;
        if(forward_one(m,&ds.examples[i],&c,err,sizeof(err))!=0){ fprintf(stderr,"forward %s\n",err); grad_free(m,&grad); dataset_free(&ds); sjev_model_free(m); return 1; }
        loss_sum += c.loss;
        printf("[");
        for(int k=0;k<c.N;k++) printf("%.8f%s", c.logits[k], k+1<c.N?",":"");
        printf("]%s", i+1<ds.n?",":"");
        if(backward_one(m,&c,&grad,err,sizeof(err))!=0){ fprintf(stderr,"backward %s\n",err); cache_free(&c); grad_free(m,&grad); dataset_free(&ds); sjev_model_free(m); return 1; }
        cache_free(&c);
    }
    float loss = (float)(loss_sum / ds.n);
    grad_scale(m,&grad, 1.0f/(float)ds.n);
    float norm_before = grad_global_norm(m,&grad);
    struct sjev_grad grad_clipped;
    grad_alloc(m,&grad_clipped,err,sizeof(err));
    {
        size_t emb_n=(size_t)257*m->width; size_t pos_n=(size_t)m->context_tokens*m->width; size_t norm_n=m->width; size_t qkv_n=(size_t)m->rank*m->width;
        memcpy(grad_clipped.embedding, grad.embedding, emb_n*sizeof(float));
        memcpy(grad_clipped.position, grad.position, pos_n*sizeof(float));
        memcpy(grad_clipped.ln_c_w, grad.ln_c_w, norm_n*sizeof(float)); memcpy(grad_clipped.ln_c_b, grad.ln_c_b, norm_n*sizeof(float));
        memcpy(grad_clipped.ln_o_w, grad.ln_o_w, norm_n*sizeof(float)); memcpy(grad_clipped.ln_o_b, grad.ln_o_b, norm_n*sizeof(float));
        memcpy(grad_clipped.Wq, grad.Wq, qkv_n*sizeof(float)); memcpy(grad_clipped.Wk, grad.Wk, qkv_n*sizeof(float)); memcpy(grad_clipped.Wv, grad.Wv, qkv_n*sizeof(float));
    }
    grad_clip(m,&grad_clipped,1.0f);
    float norm_after = grad_global_norm(m,&grad_clipped);
    struct adam_state adam;
    adam_alloc(m,&adam,0.002f,0.9f,0.999f,1e-8f,1e-4f);
    adam_step(m,&grad_clipped,&adam);
    printf("], \"loss\": %.8f, \"grad_norm_before\": %.8f, \"grad_norm_after\": %.8f, \"grads\": {", loss, norm_before, norm_after);
    printf("\"embedding\": ["); for(size_t i=0;i<(size_t)257*m->width;i++) printf("%.8f%s", grad.embedding[i], i+1<(size_t)257*m->width?",":""); printf("],");
    printf("\"position\": ["); for(size_t i=0;i<(size_t)m->context_tokens*m->width;i++) printf("%.8f%s", grad.position[i], i+1<(size_t)m->context_tokens*m->width?",":""); printf("],");
    printf("\"ln_c_w\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", grad.ln_c_w[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_c_b\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", grad.ln_c_b[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_o_w\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", grad.ln_o_w[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_o_b\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", grad.ln_o_b[i], i+1<m->width?",":""); printf("],");
    printf("\"Wq\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", grad.Wq[i], i+1<(size_t)m->rank*m->width?",":""); printf("],");
    printf("\"Wk\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", grad.Wk[i], i+1<(size_t)m->rank*m->width?",":""); printf("],");
    printf("\"Wv\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", grad.Wv[i], i+1<(size_t)m->rank*m->width?",":""); printf("]");
    printf("}, \"params_after\": {");
    printf("\"embedding\": ["); for(size_t i=0;i<(size_t)257*m->width;i++) printf("%.8f%s", m->embedding[i], i+1<(size_t)257*m->width?",":""); printf("],");
    printf("\"position\": ["); for(size_t i=0;i<(size_t)m->context_tokens*m->width;i++) printf("%.8f%s", m->position[i], i+1<(size_t)m->context_tokens*m->width?",":""); printf("],");
    printf("\"ln_c_w\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", m->ln_c_w[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_c_b\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", m->ln_c_b[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_o_w\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", m->ln_o_w[i], i+1<m->width?",":""); printf("],");
    printf("\"ln_o_b\": ["); for(size_t i=0;i<m->width;i++) printf("%.8f%s", m->ln_o_b[i], i+1<m->width?",":""); printf("],");
    printf("\"Wq\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", m->Wq[i], i+1<(size_t)m->rank*m->width?",":""); printf("],");
    printf("\"Wk\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", m->Wk[i], i+1<(size_t)m->rank*m->width?",":""); printf("],");
    printf("\"Wv\": ["); for(size_t i=0;i<(size_t)m->rank*m->width;i++) printf("%.8f%s", m->Wv[i], i+1<(size_t)m->rank*m->width?",":""); printf("]");
    printf("}}\n");
    grad_free(m,&grad); grad_free(m,&grad_clipped); adam_free(m,&adam); dataset_free(&ds); sjev_model_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if(argc<2){ print_usage(argv[0]); return 1; }
    const char *cmd=argv[1];
    if(strcmp(cmd,"predict")==0) return cmd_predict(argc-2, argv+2);
    else if(strcmp(cmd,"train")==0) return cmd_train(argc-2, argv+2);
    else if(strcmp(cmd,"eval")==0) return cmd_eval(argc-2, argv+2);
    else if(strcmp(cmd,"grad-check")==0) return cmd_grad_check(argc-2, argv+2);
    else if(strcmp(cmd,"--help")==0||strcmp(cmd,"-h")==0){ print_usage(argv[0]); return 0; }
    else { fprintf(stderr,"unknown command %s\n",cmd); print_usage(argv[0]); return 1; }
}
