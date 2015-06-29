/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 2.0.11
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package lbfgsb.jniwrapper;

public class lbfgsb {
  private long swigCPtr;
  protected boolean swigCMemOwn;

  protected lbfgsb(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(lbfgsb obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        lbfgsb_wrapperJNI.delete_lbfgsb(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setN(int value) {
    lbfgsb_wrapperJNI.lbfgsb_n_set(swigCPtr, this, value);
  }

  public int getN() {
    return lbfgsb_wrapperJNI.lbfgsb_n_get(swigCPtr, this);
  }

  public void setM(int value) {
    lbfgsb_wrapperJNI.lbfgsb_m_set(swigCPtr, this, value);
  }

  public int getM() {
    return lbfgsb_wrapperJNI.lbfgsb_m_get(swigCPtr, this);
  }

  public void setX(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_x_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getX() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_x_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public void setL(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_l_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getL() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_l_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public void setU(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_u_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getU() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_u_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public void setNbd(SWIGTYPE_p_int value) {
    lbfgsb_wrapperJNI.lbfgsb_nbd_set(swigCPtr, this, SWIGTYPE_p_int.getCPtr(value));
  }

  public SWIGTYPE_p_int getNbd() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_nbd_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_int(cPtr, false);
  }

  public void setF(double value) {
    lbfgsb_wrapperJNI.lbfgsb_f_set(swigCPtr, this, value);
  }

  public double getF() {
    return lbfgsb_wrapperJNI.lbfgsb_f_get(swigCPtr, this);
  }

  public void setG(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_g_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getG() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_g_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public void setFactr(double value) {
    lbfgsb_wrapperJNI.lbfgsb_factr_set(swigCPtr, this, value);
  }

  public double getFactr() {
    return lbfgsb_wrapperJNI.lbfgsb_factr_get(swigCPtr, this);
  }

  public void setPgtol(double value) {
    lbfgsb_wrapperJNI.lbfgsb_pgtol_set(swigCPtr, this, value);
  }

  public double getPgtol() {
    return lbfgsb_wrapperJNI.lbfgsb_pgtol_get(swigCPtr, this);
  }

  public void setWa(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_wa_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getWa() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_wa_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public void setIwa(SWIGTYPE_p_int value) {
    lbfgsb_wrapperJNI.lbfgsb_iwa_set(swigCPtr, this, SWIGTYPE_p_int.getCPtr(value));
  }

  public SWIGTYPE_p_int getIwa() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_iwa_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_int(cPtr, false);
  }

  public void setTask(String value) {
    lbfgsb_wrapperJNI.lbfgsb_task_set(swigCPtr, this, value);
  }

  public String getTask() {
    return lbfgsb_wrapperJNI.lbfgsb_task_get(swigCPtr, this);
  }

  public void setIprint(int value) {
    lbfgsb_wrapperJNI.lbfgsb_iprint_set(swigCPtr, this, value);
  }

  public int getIprint() {
    return lbfgsb_wrapperJNI.lbfgsb_iprint_get(swigCPtr, this);
  }

  public void setCsave(String value) {
    lbfgsb_wrapperJNI.lbfgsb_csave_set(swigCPtr, this, value);
  }

  public String getCsave() {
    return lbfgsb_wrapperJNI.lbfgsb_csave_get(swigCPtr, this);
  }

  public void setLsave(SWIGTYPE_p_int value) {
    lbfgsb_wrapperJNI.lbfgsb_lsave_set(swigCPtr, this, SWIGTYPE_p_int.getCPtr(value));
  }

  public SWIGTYPE_p_int getLsave() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_lsave_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_int(cPtr, false);
  }

  public void setIsave(SWIGTYPE_p_int value) {
    lbfgsb_wrapperJNI.lbfgsb_isave_set(swigCPtr, this, SWIGTYPE_p_int.getCPtr(value));
  }

  public SWIGTYPE_p_int getIsave() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_isave_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_int(cPtr, false);
  }

  public void setDsave(SWIGTYPE_p_double value) {
    lbfgsb_wrapperJNI.lbfgsb_dsave_set(swigCPtr, this, SWIGTYPE_p_double.getCPtr(value));
  }

  public SWIGTYPE_p_double getDsave() {
    long cPtr = lbfgsb_wrapperJNI.lbfgsb_dsave_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public lbfgsb() {
    this(lbfgsb_wrapperJNI.new_lbfgsb(), true);
  }

}