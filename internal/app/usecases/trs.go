package usecases

import (
	"context"

	"github.com/BaldiSlayer/rofl-lab1/internal/app/formalizeclient"
	"github.com/BaldiSlayer/rofl-lab1/internal/app/interpretclient"
	"github.com/BaldiSlayer/rofl-lab1/pkg/trsparser"
)

type TrsUseCases struct {
	parser    *trsparser.Parser
	interpret *interpretclient.Interpreter
	formalize *formalizeclient.Formalizer
}

func New() (*TrsUseCases, error) {
	interpreter, err := interpretclient.NewInterpreter()
	if err != nil {
		return nil, err
	}

	formalizer, err := formalizeclient.NewFormalizer()
	if err != nil {
		return nil, err
	}

	return &TrsUseCases{
		parser:    trsparser.NewParser(),
		interpret: interpreter,
		formalize: formalizer,
	}, nil
}

func (uc *TrsUseCases) ExtractFormalTrs(ctx context.Context, request string) (trsparser.Trs, string, error) {
	formalizedTrs, err := uc.formalize.Formalize(ctx, request)
	if err != nil {
		return trsparser.Trs{}, "", err
	}

	trs, err := uc.parser.Parse(formalizedTrs)
	if err != nil {
		return trsparser.Trs{}, formalizedTrs, err
	}

	return *trs, formalizedTrs, nil
}

func (uc *TrsUseCases) FixFormalTrs(ctx context.Context, request, formalTrs, errorDescription string) (trsparser.Trs, string, error) {
	formalizedTrs, err := uc.formalize.FixFormalized(ctx, request, formalTrs, errorDescription)
	if err != nil {
		return trsparser.Trs{}, "", err
	}

	trs, err := uc.parser.Parse(formalizedTrs)
	if err != nil {
		return trsparser.Trs{}, formalizedTrs, err
	}

	return *trs, formalizedTrs, nil
}

func (uc *TrsUseCases) InterpretFormalTrs(ctx context.Context, trs trsparser.Trs) (string, error) {
	return uc.interpret.Interpret(ctx, trs)
}
