import { useState } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ErrorBoundary from '../components/ErrorBoundary';

// A component that throws on render when the `shouldThrow` prop is true
function Bomb({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error('Test error message');
  return <p>All good</p>;
}

// Suppress console.error noise from intentional throws in tests
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => {
  vi.restoreAllMocks();
});

describe('ErrorBoundary', () => {
  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow={false} />
      </ErrorBoundary>
    );
    expect(screen.getByText('All good')).toBeInTheDocument();
  });

  it('renders fallback UI when a child throws', () => {
    render(
      <ErrorBoundary>
        <Bomb shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to home' })).toBeInTheDocument();
  });

  it('renders custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<p>Custom fallback</p>}>
        <Bomb shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
  });

  it('resets to children after clicking Try again', async () => {
    const user = userEvent.setup();

    // Wrapper lets us stop the throw before resetting the boundary
    function BombWrapper() {
      const [shouldThrow, setShouldThrow] = useState(true);
      return (
        <div>
          <button onClick={() => setShouldThrow(false)}>stop throwing</button>
          <ErrorBoundary>
            <Bomb shouldThrow={shouldThrow} />
          </ErrorBoundary>
        </div>
      );
    }

    render(<BombWrapper />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Stop the throw, then reset the boundary — now renders the healthy child
    await user.click(screen.getByRole('button', { name: 'stop throwing' }));
    await user.click(screen.getByRole('button', { name: 'Try again' }));

    expect(screen.getByText('All good')).toBeInTheDocument();
  });
});
