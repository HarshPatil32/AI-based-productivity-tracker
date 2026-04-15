import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import axios from 'axios';
import ErrorMessage from '../components/shared/ErrorMessage';

describe('ErrorMessage', () => {
  it('renders nothing when error is null', () => {
    const { container } = render(<ErrorMessage error={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when error is undefined', () => {
    const { container } = render(<ErrorMessage error={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('extracts detail string from an axios error', () => {
    const axiosError = new axios.AxiosError(
      'Request failed',
      '400',
      undefined,
      undefined,
      {
        data: { detail: 'Username already taken' },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: { headers: {} } as any,
      }
    );

    render(<ErrorMessage error={axiosError} />);
    expect(screen.getByText('Username already taken')).toBeInTheDocument();
  });

  it('falls back to error.message for a plain Error', () => {
    render(<ErrorMessage error={new Error('Something exploded')} />);
    expect(screen.getByText('Something exploded')).toBeInTheDocument();
  });

  it('falls back to generic message for unknown error shape', () => {
    render(<ErrorMessage error={{ unexpected: true }} />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('applies a custom className', () => {
    render(<ErrorMessage error={new Error('oops')} className="my-custom-class" />);
    expect(screen.getByText('oops')).toHaveClass('my-custom-class');
  });
});
