// Package formalizeclient provides primitives to interact with the openapi HTTP API.
//
// Code generated by github.com/oapi-codegen/oapi-codegen/v2 version v2.3.0 DO NOT EDIT.
package formalizeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// FormalizeRequest defines model for FormalizeRequest.
type FormalizeRequest struct {
	Trs string `json:"trs"`
}

// FormalizeResult defines model for FormalizeResult.
type FormalizeResult struct {
	FormalTrs string `json:"formalTrs"`
}

// TrsFormalizeJSONRequestBody defines body for TrsFormalize for application/json ContentType.
type TrsFormalizeJSONRequestBody = FormalizeRequest

// RequestEditorFn  is the function signature for the RequestEditor callback function
type RequestEditorFn func(ctx context.Context, req *http.Request) error

// Doer performs HTTP requests.
//
// The standard http.Client implements this interface.
type HttpRequestDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Client which conforms to the OpenAPI3 specification for this service.
type Client struct {
	// The endpoint of the server conforming to this interface, with scheme,
	// https://api.deepmap.com for example. This can contain a path relative
	// to the server, such as https://api.deepmap.com/dev-test, and all the
	// paths in the swagger spec will be appended to the server.
	Server string

	// Doer for performing requests, typically a *http.Client with any
	// customized settings, such as certificate chains.
	Client HttpRequestDoer

	// A list of callbacks for modifying requests which are generated before sending over
	// the network.
	RequestEditors []RequestEditorFn
}

// ClientOption allows setting custom parameters during construction
type ClientOption func(*Client) error

// Creates a new Client, with reasonable defaults
func NewClient(server string, opts ...ClientOption) (*Client, error) {
	// create a client with sane default values
	client := Client{
		Server: server,
	}
	// mutate client and add all optional params
	for _, o := range opts {
		if err := o(&client); err != nil {
			return nil, err
		}
	}
	// ensure the server URL always has a trailing slash
	if !strings.HasSuffix(client.Server, "/") {
		client.Server += "/"
	}
	// create httpClient, if not already present
	if client.Client == nil {
		client.Client = &http.Client{}
	}
	return &client, nil
}

// WithHTTPClient allows overriding the default Doer, which is
// automatically created using http.Client. This is useful for tests.
func WithHTTPClient(doer HttpRequestDoer) ClientOption {
	return func(c *Client) error {
		c.Client = doer
		return nil
	}
}

// WithRequestEditorFn allows setting up a callback function, which will be
// called right before sending the request. This can be used to mutate the request.
func WithRequestEditorFn(fn RequestEditorFn) ClientOption {
	return func(c *Client) error {
		c.RequestEditors = append(c.RequestEditors, fn)
		return nil
	}
}

// The interface specification for the client above.
type ClientInterface interface {
	// Healthcheck request
	Healthcheck(ctx context.Context, reqEditors ...RequestEditorFn) (*http.Response, error)

	// TrsFormalizeWithBody request with any body
	TrsFormalizeWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error)

	TrsFormalize(ctx context.Context, body TrsFormalizeJSONRequestBody, reqEditors ...RequestEditorFn) (*http.Response, error)
}

func (c *Client) Healthcheck(ctx context.Context, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewHealthcheckRequest(c.Server)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) TrsFormalizeWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewTrsFormalizeRequestWithBody(c.Server, contentType, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) TrsFormalize(ctx context.Context, body TrsFormalizeJSONRequestBody, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewTrsFormalizeRequest(c.Server, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

// NewHealthcheckRequest generates requests for Healthcheck
func NewHealthcheckRequest(server string) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/ping")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("GET", queryURL.String(), nil)
	if err != nil {
		return nil, err
	}

	return req, nil
}

// NewTrsFormalizeRequest calls the generic TrsFormalize builder with application/json body
func NewTrsFormalizeRequest(server string, body TrsFormalizeJSONRequestBody) (*http.Request, error) {
	var bodyReader io.Reader
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	bodyReader = bytes.NewReader(buf)
	return NewTrsFormalizeRequestWithBody(server, "application/json", bodyReader)
}

// NewTrsFormalizeRequestWithBody generates requests for TrsFormalize with any type of body
func NewTrsFormalizeRequestWithBody(server string, contentType string, body io.Reader) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/trs/formalize")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", queryURL.String(), body)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", contentType)

	return req, nil
}

func (c *Client) applyEditors(ctx context.Context, req *http.Request, additionalEditors []RequestEditorFn) error {
	for _, r := range c.RequestEditors {
		if err := r(ctx, req); err != nil {
			return err
		}
	}
	for _, r := range additionalEditors {
		if err := r(ctx, req); err != nil {
			return err
		}
	}
	return nil
}

// ClientWithResponses builds on ClientInterface to offer response payloads
type ClientWithResponses struct {
	ClientInterface
}

// NewClientWithResponses creates a new ClientWithResponses, which wraps
// Client with return type handling
func NewClientWithResponses(server string, opts ...ClientOption) (*ClientWithResponses, error) {
	client, err := NewClient(server, opts...)
	if err != nil {
		return nil, err
	}
	return &ClientWithResponses{client}, nil
}

// WithBaseURL overrides the baseURL.
func WithBaseURL(baseURL string) ClientOption {
	return func(c *Client) error {
		newBaseURL, err := url.Parse(baseURL)
		if err != nil {
			return err
		}
		c.Server = newBaseURL.String()
		return nil
	}
}

// ClientWithResponsesInterface is the interface specification for the client with responses above.
type ClientWithResponsesInterface interface {
	// HealthcheckWithResponse request
	HealthcheckWithResponse(ctx context.Context, reqEditors ...RequestEditorFn) (*HealthcheckResponse, error)

	// TrsFormalizeWithBodyWithResponse request with any body
	TrsFormalizeWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*TrsFormalizeResponse, error)

	TrsFormalizeWithResponse(ctx context.Context, body TrsFormalizeJSONRequestBody, reqEditors ...RequestEditorFn) (*TrsFormalizeResponse, error)
}

type HealthcheckResponse struct {
	Body         []byte
	HTTPResponse *http.Response
}

// Status returns HTTPResponse.Status
func (r HealthcheckResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r HealthcheckResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

type TrsFormalizeResponse struct {
	Body         []byte
	HTTPResponse *http.Response
	JSON200      *FormalizeResult
}

// Status returns HTTPResponse.Status
func (r TrsFormalizeResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r TrsFormalizeResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

// HealthcheckWithResponse request returning *HealthcheckResponse
func (c *ClientWithResponses) HealthcheckWithResponse(ctx context.Context, reqEditors ...RequestEditorFn) (*HealthcheckResponse, error) {
	rsp, err := c.Healthcheck(ctx, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseHealthcheckResponse(rsp)
}

// TrsFormalizeWithBodyWithResponse request with arbitrary body returning *TrsFormalizeResponse
func (c *ClientWithResponses) TrsFormalizeWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*TrsFormalizeResponse, error) {
	rsp, err := c.TrsFormalizeWithBody(ctx, contentType, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseTrsFormalizeResponse(rsp)
}

func (c *ClientWithResponses) TrsFormalizeWithResponse(ctx context.Context, body TrsFormalizeJSONRequestBody, reqEditors ...RequestEditorFn) (*TrsFormalizeResponse, error) {
	rsp, err := c.TrsFormalize(ctx, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseTrsFormalizeResponse(rsp)
}

// ParseHealthcheckResponse parses an HTTP response from a HealthcheckWithResponse call
func ParseHealthcheckResponse(rsp *http.Response) (*HealthcheckResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &HealthcheckResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	return response, nil
}

// ParseTrsFormalizeResponse parses an HTTP response from a TrsFormalizeWithResponse call
func ParseTrsFormalizeResponse(rsp *http.Response) (*TrsFormalizeResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &TrsFormalizeResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	switch {
	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 200:
		var dest FormalizeResult
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON200 = &dest

	}

	return response, nil
}
