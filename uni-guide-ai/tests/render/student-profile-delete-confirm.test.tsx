import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// Only the profile hook is faked: the form reads profile state from it.
vi.mock('@/hooks/useStudentProfile', () => ({
  useStudentProfile: () =>
    new Proxy(
      { profile: {} },
      { get: (target, key) => (key in target ? (target as Record<string, unknown>)[key as string] : vi.fn()) },
    ),
}));

import { StudentProfileForm } from '@/components/student/StudentProfileForm';

describe('StudentProfileForm delete confirmation', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  let confirmSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    confirmSpy.mockRestore();
  });

  it('asks in an AlertDialog instead of window.confirm and deletes only on confirm', async () => {
    render(<StudentProfileForm />);
    fireEvent.click(screen.getByRole('button', { name: 'Xóa hồ sơ' }));

    const dialog = await screen.findByRole('alertdialog');
    expect(dialog).toHaveTextContent('Xóa hồ sơ thí sinh?');
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Hủy' }));
    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(fetchMock).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Xóa hồ sơ' }));
    const reopened = await screen.findByRole('alertdialog');
    fireEvent.click(within(reopened).getByRole('button', { name: 'Xóa hồ sơ' }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith('/v2/student-profile', { method: 'DELETE' }),
    );
  });
});
