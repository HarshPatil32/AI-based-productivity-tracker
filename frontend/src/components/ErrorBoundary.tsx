import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (import.meta.env.DEV) {
      console.error('[ErrorBoundary]', error, info.componentStack);
    }
  }

  resetError = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    if (this.props.fallback) {
      return this.props.fallback;
    }

    const { error } = this.state;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg border border-gray-200 p-8 max-w-md w-full text-center space-y-4">
          <h1 className="text-xl font-semibold text-gray-900">Something went wrong</h1>
          <p className="text-sm text-gray-500">
            An unexpected error occurred. You can try again or return to the home page.
          </p>
          {import.meta.env.DEV && error && (
            <details className="text-left text-xs bg-gray-50 rounded p-3 border border-gray-200">
              <summary className="cursor-pointer font-medium text-gray-700 mb-1">
                Error details
              </summary>
              <pre className="whitespace-pre-wrap break-words text-red-600 mt-2">
                {error.message}
                {error.stack ? `\n\n${error.stack}` : ''}
              </pre>
            </details>
          )}
          <div className="flex gap-3 justify-center pt-2">
            <button
              onClick={this.resetError}
              className="px-4 py-2 rounded-md bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 transition-colors"
            >
              Try again
            </button>
            <a
              href="/"
              className="px-4 py-2 rounded-md border border-gray-300 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Go to home
            </a>
          </div>
        </div>
      </div>
    );
  }
}
